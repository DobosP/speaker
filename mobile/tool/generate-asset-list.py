#!/usr/bin/env python3
"""Rewrite the `assets:` block in pubspec.yaml to list every non-empty
subfolder under ./assets/.

Flutter does not bundle asset folders recursively, so each directory that
contains files (e.g. the model dirs and every espeak-ng-data locale subdir)
must be listed explicitly. This script regenerates that list after the model
archives have been extracted.

Assumes `  assets:` is the final key in pubspec.yaml (see pubspec.yaml note).
"""
import os

PUBSPEC = "pubspec.yaml"
ASSETS = "assets"


def non_empty_subfolders():
    folders = set()
    for root, _dirs, files in os.walk(ASSETS):
        if files:
            rel = root.replace("\\", "/").rstrip("/")
            folders.add(f"    - {rel}/")
    return sorted(folders)


def main():
    with open(PUBSPEC, encoding="utf-8") as f:
        lines = f.readlines()

    cut = next((i for i, ln in enumerate(lines) if ln.rstrip() == "  assets:"), None)
    if cut is None:
        raise SystemExit("Could not find '  assets:' in pubspec.yaml")

    head = lines[: cut + 1]
    folders = non_empty_subfolders()
    if not folders:
        raise SystemExit("No asset folders found under ./assets/ — run download-models.sh first")

    with open(PUBSPEC, "w", encoding="utf-8") as f:
        f.writelines(head)
        f.write("\n".join(folders) + "\n")

    print(f"Wrote {len(folders)} asset folder entries to {PUBSPEC}")


if __name__ == "__main__":
    main()
