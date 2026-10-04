#!/usr/bin/env python3
"""Generate Flutter assets with the exact active ASR tuple and optional Whisper.

Model directories contain unused precisions and examples. ASR/Whisper entries
are explicit files; other assets keep their nested directory layout (including
TTS espeak-ng-data). Cached files are never removed. `assets:` stays the final
pubspec key.
"""

import argparse
import os
from pathlib import Path

PUBSPEC = "pubspec.yaml"
ASSETS = "assets"
ASR = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
OPTIONAL_WHISPER = "sherpa-onnx-whisper-base.en"
ASR_FILES = (
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
    "tokens.txt",
)
WHISPER_FILES = (
    "base.en-encoder.int8.onnx",
    "base.en-decoder.int8.onnx",
    "base.en-tokens.txt",
)


def _is_rights_information(name: str) -> bool:
    lowered = name.casefold()
    return any(
        lowered == stem
        or any(lowered.startswith(stem + separator) for separator in (".", "-", "_"))
        for stem in ("license", "copying", "notice", "readme")
    )


def _model_file_entries(family: str, filenames: tuple[str, ...]) -> set[str]:
    directory = Path(ASSETS) / family
    paths = [directory / name for name in filenames]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise SystemExit(
            f"Required mobile model tuple is incomplete: {family}; run download-models.sh"
        )
    paths.extend(
        path
        for path in directory.iterdir()
        if path.is_file() and _is_rights_information(path.name)
    )
    return {f"    - {path.as_posix()}" for path in paths}


def asset_entries(*, with_whisper: bool = False) -> list[str]:
    entries = _model_file_entries(ASR, ASR_FILES)
    if with_whisper:
        entries.update(_model_file_entries(OPTIONAL_WHISPER, WHISPER_FILES))
    for root, dirs, files in os.walk(ASSETS):
        if root == ASSETS:
            # Explicit file entries replace whole-model directory entries.
            # Optional caches remain on disk and cannot enter a default bundle.
            dirs[:] = [name for name in dirs if name not in {ASR, OPTIONAL_WHISPER}]
        if files:
            rel = root.replace("\\", "/").rstrip("/")
            entries.add(f"    - {rel}/")
    return sorted(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle exact mobile model files; unused Whisper is opt-in."
    )
    parser.add_argument(
        "--with-whisper",
        action="store_true",
        help="also bundle the three separately downloaded optional Whisper files",
    )
    args = parser.parse_args()
    with open(PUBSPEC, encoding="utf-8") as f:
        lines = f.readlines()
    cut = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "  assets:"), None
    )
    if cut is None:
        raise SystemExit("Could not find '  assets:' in pubspec.yaml")
    entries = asset_entries(with_whisper=args.with_whisper)
    with open(PUBSPEC, "w", encoding="utf-8") as f:
        f.writelines(lines[: cut + 1])
        f.write("\n".join(entries) + "\n")
    print(f"Wrote {len(entries)} asset entries to {PUBSPEC}")


if __name__ == "__main__":
    main()
