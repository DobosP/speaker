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
        n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
        options=llm_cfg.get("options"),
    )
    main = LlamaCppLLM(paths.main_gguf, **common)
    fast = LlamaCppLLM(paths.fast_gguf, **common) if paths.fast_gguf else None
    return main, fast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-model latency benchmark.")
    parser.add_argument("--profile", default="phone", help="device profile (phone/desktop)")
    parser.add_argument("--fixtures", default="tests/fixture_audio/virtual_real_world")
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

    out_dir = Path(args.out) / args.profile

    if args.fake:
        cases = _fake_cases(args.fixtures if os.path.isdir(args.fixtures) else None, args.limit)
        samples = runner.run_fake(cases)
    else:
        from .models import fetch_models

        config = _apply_device_profile(_load_config(), args.profile)
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        paths = fetch_models(args.cache_dir, token=token, manifest_path=args.models_manifest)
        sherpa_cfg = _build_sherpa_config(config, paths.sherpa_overrides())
        main_llm, fast_llm = _build_llms(config, paths)
        fixtures = runner.discover_fixtures(args.fixtures, limit=args.limit)
        if not fixtures:
            raise SystemExit(f"No fixtures in {args.fixtures!r}")
        samples = runner.run_real(
            fixtures, sherpa_cfg, main_llm, fast_llm, stream_tts=args.stream_tts
        )

    index = report.write_reports(out_dir, args.profile, samples)
    print(f"[bench] wrote {index}")
    if args.md_summary:
        with open(args.md_summary, "a", encoding="utf-8") as fh:
            fh.write(report.markdown_summary(args.profile, samples))
    # Echo the calibration table to stdout for quick reading.
    print(report.markdown_summary(args.profile, samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
