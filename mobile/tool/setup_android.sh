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

# Some plugins (e.g. audioplayers) pull AndroidX libs that require compileSdk
# >= 34, but `flutter create` may default to 33. Force a high enough compileSdk.
python3 - <<'PY'
import os
import re

candidates = ["android/app/build.gradle.kts", "android/app/build.gradle"]
path = next((p for p in candidates if os.path.exists(p)), None)
if path is None:
    raise SystemExit("Could not find android/app/build.gradle[.kts]")

with open(path, encoding="utf-8") as f:
    text = f.read()

# Normalize whatever compileSdk line is present to a fixed value (valid in
# both Groovy and Kotlin DSL).
new, n = re.subn(r"compileSdk(?:Version)?\s*=?\s*[^\n]+", "compileSdk = 34", text)
if n == 0:
    raise SystemExit(f"No compileSdk declaration found in {path}")

with open(path, "w", encoding="utf-8") as f:
    f.write(new)
print(f"Set compileSdk = 34 in {path} ({n} occurrence(s))")
PY
