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
    '    <uses-permission android:name="android.permission.INTERNET" />\n'
)

changed = False
if "android.permission.RECORD_AUDIO" not in text:
    text = text.replace("    <application", perms + "    <application", 1)
    changed = True
    print("Patched AndroidManifest.xml with RECORD_AUDIO + INTERNET permissions")
else:
    print("RECORD_AUDIO already present, no permission patch needed")

# MediaPipe's GPU LLM backend (flutter_gemma) dlopen()s OpenCL; declaring the
# native libs as optional lets it use the GPU where present and fall back to
# CPU elsewhere. These must live inside <application>.
opencl = (
    '        <uses-native-library android:name="libOpenCL.so" android:required="false" />\n'
    '        <uses-native-library android:name="libOpenCL-car.so" android:required="false" />\n'
    '        <uses-native-library android:name="libOpenCL-pixel.so" android:required="false" />\n'
)
if "libOpenCL.so" not in text:
    text = text.replace("    </application>", opencl + "    </application>", 1)
    changed = True
    print("Patched AndroidManifest.xml with OpenCL uses-native-library entries")
else:
    print("OpenCL uses-native-library already present, no patch needed")

if changed:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
PY

# Plugins (record_android, audioplayers) pull AndroidX libs that require
# compileSdk 36, but `flutter create` may default lower. Force compileSdk 36.
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
text, n = re.subn(r"compileSdk(?:Version)?\s*=?\s*[^\n]+", "compileSdk = 36", text)
if n == 0:
    raise SystemExit(f"No compileSdk declaration found in {path}")

# flutter_gemma / MediaPipe GenAI require minSdk >= 24; flutter's default
# (flutter.minSdkVersion) is lower, so pin it.
text, m = re.subn(r"minSdk(?:Version)?\s*=?\s*[^\n]+", "minSdk = 24", text)
if m == 0:
    raise SystemExit(f"No minSdk declaration found in {path}")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"Set compileSdk = 36 ({n}x) and minSdk = 24 ({m}x) in {path}")
PY

# Flutter plugin sub-projects (record_android, audioplayers_android, ...) keep
# their OWN compileSdk and ignore the app module's. Force every Android
# sub-project to compileSdk 36 via the root Gradle file.
python3 - <<'PY'
import os

groovy = "android/build.gradle"
kts = "android/build.gradle.kts"

if os.path.exists(kts):
    path, block = kts, '''
subprojects {
    if (state.executed) {
        extensions.findByName("android")?.withGroovyBuilder { "compileSdkVersion"(36) }
    } else {
        afterEvaluate {
            extensions.findByName("android")?.withGroovyBuilder { "compileSdkVersion"(36) }
        }
    }
}
'''
elif os.path.exists(groovy):
    path, block = groovy, '''
subprojects {
    if (project.state.executed) {
        if (project.extensions.findByName("android") != null) { project.android.compileSdkVersion 36 }
    } else {
        afterEvaluate {
            if (project.extensions.findByName("android") != null) { project.android.compileSdkVersion 36 }
        }
    }
}
'''
else:
    raise SystemExit("Could not find root android/build.gradle[.kts]")

with open(path, encoding="utf-8") as f:
    text = f.read()

if "forced compileSdk for all subprojects" not in text:
    text += "\n// forced compileSdk for all subprojects\n" + block
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Forced compileSdk 36 on all subprojects in {path}")
else:
    print("Subproject compileSdk override already present")
PY
