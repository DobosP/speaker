#!/usr/bin/env bash
# Materialize the Android platform project (gradle wrapper, manifest, etc.)
# for this Flutter app, then patch the manifest to request microphone access.
#
# We generate the platform files at build time instead of committing them, so
# the repo stays free of the gradle wrapper jar and version-specific scaffolding.
set -euo pipefail

cd "$(dirname "$0")/.."

flutter create --org com.k2fsa.sherpa --project-name speaker_mobile \
  --platforms=android .

MANIFEST=android/app/src/main/AndroidManifest.xml

python3 - "$MANIFEST" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    text = f.read()

perms = (
    '    <uses-permission android:name="android.permission.RECORD_AUDIO" />\n'
    '    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />\n'
)

if "android.permission.RECORD_AUDIO" not in text:
    text = text.replace("    <application", perms + "    <application", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Patched AndroidManifest.xml with RECORD_AUDIO permission")
else:
    print("RECORD_AUDIO already present, no patch needed")
PY
