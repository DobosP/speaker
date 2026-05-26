#!/usr/bin/env python3
"""
Benchmark open-weight ASR models on microphone clips from recorded sessions.

Uses ``recordings/session_*/turn_*/mic_16k.npy`` (16 kHz float32) from the latest
sessions and the project's ``transcribe_audio`` API (same stack as main.py).

Default **20** model configurations: Whisper-class sizes (English + multilingual
via explicit ``Systran/faster-whisper-*`` Hub ids where ids would collide with
whisper.cpp), Moonshine, whisper.cpp, and multiple large tiers (**large-v2**,
**distil-large-v3**, **distil-large-v3.5**, turbo).

Each entry carries a coarse **size** label (``xs`` / ``sm`` / ``md`` / ``lg``) for
``--size`` filtering. Use ``--only`` to pick exact model ids.

Requires **faster-whisper ≥ 1.2** so keys like ``distil-large-v3.5`` resolve.

Recordings often omit reference transcripts; this benchmark reports **RTF**
(real‑time factor = CPU seconds / audio seconds) and **hypothesis text** for
qualitative comparison. Optional ``--wer-reference`` supplies one gold string
for a quick WER when ``jiwer`` is installed.

Example::

    python benchmarks/benchmark_session_stt.py --latest-sessions 4 --json-out benchmarks/last_stt_benchmark.json

Environment: runs offline after models are cached; first run downloads weights.
Peak RAM can be large when loading ``large-v3`` class models sequentially.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stt import transcribe_audio  # noqa: E402


# 20 presets — unique ``id`` values (``--only``); multilingual Whisper tiers use
# explicit Hub ids so they do not collide with whisper.cpp ``tiny``/``small``/``medium``.
DEFAULT_MODELS: list[dict[str, str]] = [
    # faster-whisper — English + multilingual + distilled + legacy large
    {"id": "tiny.en", "type": "whisper", "size": "xs", "family": "Whisper tiny.en"},
    {"id": "Systran/faster-whisper-tiny", "type": "whisper", "size": "xs", "family": "Whisper tiny (multilingual CT2)"},
    {"id": "base", "type": "whisper", "size": "sm", "family": "Whisper base"},
    {"id": "base.en", "type": "whisper", "size": "sm", "family": "Whisper base.en"},
    {"id": "small.en", "type": "whisper", "size": "sm", "family": "Whisper small.en"},
    {"id": "Systran/faster-whisper-small", "type": "whisper", "size": "sm", "family": "Whisper small (multilingual CT2)"},
    {"id": "distil-small.en", "type": "whisper", "size": "sm", "family": "Distil-Whisper small.en"},
    {"id": "medium.en", "type": "whisper", "size": "md", "family": "Whisper medium.en"},
    {"id": "Systran/faster-whisper-medium", "type": "whisper", "size": "md", "family": "Whisper medium (multilingual CT2)"},
    {"id": "distil-medium.en", "type": "whisper", "size": "md", "family": "Distil-Whisper medium.en"},
    {"id": "large-v3-turbo", "type": "whisper", "size": "lg", "family": "Whisper large-v3 turbo (mobiuslabs CT2)"},
    {"id": "large-v3", "type": "whisper", "size": "lg", "family": "Whisper large-v3 (Systran CT2)"},
    {"id": "large-v2", "type": "whisper", "size": "lg", "family": "Whisper large-v2 (Systran CT2)"},
    {"id": "distil-large-v3.5", "type": "whisper", "size": "lg", "family": "Distil-Whisper large v3.5 (CT2)"},
    {"id": "distil-large-v3", "type": "whisper", "size": "lg", "family": "Distil-Whisper large v3 (CT2)"},
    # Moonshine ONNX
    {"id": "moonshine:tiny", "type": "moonshine", "size": "xs", "family": "Moonshine ONNX tiny"},
    {"id": "moonshine:base", "type": "moonshine", "size": "sm", "family": "Moonshine ONNX base"},
    # whisper.cpp GGML
    {"id": "tiny", "type": "whispercpp", "size": "xs", "family": "whisper.cpp GGML tiny"},
    {"id": "small", "type": "whispercpp", "size": "sm", "family": "whisper.cpp GGML small"},
    {"id": "medium", "type": "whispercpp", "size": "md", "family": "whisper.cpp GGML medium"},
]


@dataclass
class TurnClip:
    session_id: str
    turn_index: int
    path: Path
    audio_sec: float
    samples: int


def _reset_stt_singletons() -> None:
    """Best-effort unload so switching families does not accumulate VRAM/RAM."""
    import utils.stt as st

    for attr in ("_stt_model", "_moonshine_stt", "_streaming_stt"):
        old = getattr(st, attr, None)
        setattr(st, attr, None)
        if old is not None:
            try:
                if hasattr(old, "model"):
                    del old.model
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _session_dirs_sorted(recordings: Path) -> list[Path]:
    dirs = [p for p in recordings.glob("session_*") if p.is_dir()]
    return sorted(
        dirs,
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )


def _collect_turns(
    recordings: Path,
    *,
    latest_sessions: int,
    min_samples: int,
    max_turns_total: int | None,
) -> list[TurnClip]:
    clips: list[TurnClip] = []
    seen = 0
    for sess_dir in _session_dirs_sorted(recordings)[:latest_sessions]:
        meta_path = sess_dir / "metadata.json"
        sid = sess_dir.name
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    sid = json.load(f).get("session_id", sid)
            except Exception:
                pass
        for mic in sorted(sess_dir.glob("turn_*/mic_16k.npy")):
            try:
                audio = np.load(mic)
            except Exception:
                continue
            n = int(np.asarray(audio).size)
            if n < min_samples:
                continue
            parts = mic.parent.name.split("_")
            idx = int(parts[-1]) if parts[-1].isdigit() else -1
            clips.append(
                TurnClip(
                    session_id=sid,
                    turn_index=idx,
                    path=mic,
                    audio_sec=n / 16000.0,
                    samples=n,
                )
            )
            seen += 1
            if max_turns_total is not None and seen >= max_turns_total:
                return clips
    return clips


def _maybe_wer(ref: str, hyp: str) -> float | None:
    if not ref.strip():
        return None
    try:
        import jiwer

        return float(jiwer.wer(ref.lower().strip(), hyp.lower().strip()))
    except Exception:
        return None


def run_benchmark(
    clips: list[TurnClip],
    models: list[dict[str, str]],
    *,
    n_threads: int,
    wer_reference: str | None,
) -> dict:
    rows: list[dict] = []
    total_audio_sec = sum(c.audio_sec for c in clips)

    for spec in models:
        mid = spec["id"]
        mtype = spec["type"]
        family = spec.get("family", mid)
        _reset_stt_singletons()
        t_load_start = time.perf_counter()
        # Prime load with empty audio (fast path still touches model init).
        try:
            transcribe_audio(
                np.zeros(1600, dtype=np.float32),
                model_id=mid,
                model_type=mtype,
                n_threads=n_threads,
            )
        except Exception as e:
            rows.append(
                {
                    "model_id": mid,
                    "model_type": mtype,
                    "family": family,
                    "size": spec.get("size"),
                    "error": str(e),
                }
            )
            continue
        load_sec = time.perf_counter() - t_load_start

        transcribe_sec = 0.0
        hyps: list[str] = []
        errs: list[str] = []
        for clip in clips:
            audio = np.load(clip.path)
            t0 = time.perf_counter()
            try:
                text = transcribe_audio(
                    audio,
                    model_id=mid,
                    model_type=mtype,
                    n_threads=n_threads,
                )
                transcribe_sec += time.perf_counter() - t0
                hyps.append(text)
            except Exception as e:
                errs.append(f"{clip.path}: {e}")
                hyps.append("")

        rtf = (
            round(transcribe_sec / total_audio_sec, 4) if total_audio_sec > 0 else None
        )
        row: dict = {
            "model_id": mid,
            "model_type": mtype,
            "family": family,
            "size": spec.get("size"),
            "load_wall_sec": round(load_sec, 3),
            "transcribe_wall_sec": round(transcribe_sec, 3),
            "total_audio_sec": round(total_audio_sec, 3),
            "rtf": rtf,
            "hypotheses": [
                {
                    "session": c.session_id,
                    "turn": c.turn_index,
                    "path": str(c.path),
                    "text": hyps[i] if i < len(hyps) else "",
                }
                for i, c in enumerate(clips)
            ],
        }
        if errs:
            row["errors"] = errs
        if wer_reference and hyps:
            wers = []
            for h in hyps:
                w = _maybe_wer(wer_reference, h)
                if w is not None:
                    wers.append(w)
            if wers:
                row["wer_vs_reference"] = round(sum(wers) / len(wers), 4)
        rows.append(row)

    _reset_stt_singletons()
    return {
        "clips": [
            {
                "session_id": c.session_id,
                "turn_index": c.turn_index,
                "path": str(c.path),
                "audio_sec": c.audio_sec,
                "samples": c.samples,
            }
            for c in clips
        ],
        "models": rows,
        "n_threads": n_threads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark open STT models on session mic recordings."
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Set CUDA_VISIBLE_DEVICES= (use CPU; avoids some broken cuDNN/ONNX init crashes)",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=ROOT / "recordings",
        help="Path to recordings/ (session_*/turn_*/mic_16k.npy)",
    )
    parser.add_argument(
        "--latest-sessions",
        type=int,
        default=4,
        help="Use mic clips from the N most recently modified sessions",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8000,
        help="Skip clips shorter than this many samples (~0.5s at 16 kHz)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap total clips after filtering (default: all from selected sessions)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(4, (os.cpu_count() or 4) // 2),
        help="Thread budget for faster-whisper / whisper.cpp",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="MODEL_ID",
        default=None,
        help="Subset of default model ids (e.g. --only base small.en large-v3-turbo)",
    )
    parser.add_argument(
        "--size",
        nargs="+",
        choices=["xs", "sm", "md", "lg"],
        metavar="SZ",
        default=None,
        help="Keep only models with this coarse size tag (xs=tiny-class … lg=large-class)",
    )
    parser.add_argument(
        "--wer-reference",
        type=str,
        default=None,
        help="Optional single reference transcript for mean WER (requires jiwer)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full results as JSON",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print default model list and exit",
    )
    args = parser.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.list_models:
        for m in DEFAULT_MODELS:
            print(
                f"{m['id']}\t{m.get('size', ''):>2}\t{m['type']}\t{m.get('family', '')}"
            )
        return 0

    if not args.recordings.is_dir():
        print(f"Recordings directory not found: {args.recordings}", file=sys.stderr)
        return 2

    clips = _collect_turns(
        args.recordings,
        latest_sessions=args.latest_sessions,
        min_samples=args.min_samples,
        max_turns_total=args.max_turns,
    )
    if not clips:
        print(
            "No mic clips found. Record a session first or lower --min-samples.",
            file=sys.stderr,
        )
        return 2

    models = DEFAULT_MODELS
    if args.only:
        want = {x.strip() for x in args.only}
        models = [m for m in DEFAULT_MODELS if m["id"] in want]
        missing = want - {m["id"] for m in models}
        if missing:
            print(f"Unknown model id(s) (not in default list): {sorted(missing)}", file=sys.stderr)
            return 2
        if not models:
            print("--only matched no models.", file=sys.stderr)
            return 2

    if args.size:
        allow = set(args.size)
        models = [m for m in models if m.get("size") in allow]
        if not models:
            print("--size excluded all models.", file=sys.stderr)
            return 2

    print(
        f"Models: {len(models)}  |  Clips: {len(clips)} turns from latest "
        f"{args.latest_sessions} session(s), ~{sum(c.audio_sec for c in clips):.1f}s audio\n"
    )

    report = run_benchmark(
        clips,
        models,
        n_threads=args.threads,
        wer_reference=args.wer_reference,
    )

    # Pretty table (RTF lower is faster; <1 means faster than realtime)
    col_w = 42
    print(
        f"{'model_id':<{col_w}} {'sz':>3} {'type':<12} {'RTF':>8} {'load_s':>8} {'transc_s':>10}"
    )
    print("-" * (col_w + 38))
    for row in report["models"]:
        if "error" in row:
            mid = row["model_id"]
            if len(mid) > col_w:
                mid = mid[: col_w - 3] + "..."
            print(
                f"{mid:<{col_w}} {str(row.get('size') or ''):>3} "
                f"{row['model_type']:<12} ERROR: {row['error']}"
            )
            continue
        mid = row["model_id"]
        if len(mid) > col_w:
            mid = mid[: col_w - 3] + "..."
        print(
            f"{mid:<{col_w}} {str(row.get('size') or ''):>3} {row['model_type']:<12} "
            f"{row.get('rtf') or 0:>8.3f} {row['load_wall_sec']:>8.2f} "
            f"{row['transcribe_wall_sec']:>10.2f}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
