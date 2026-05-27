#!/usr/bin/env bash
# Fast on-device test loop: (re)install the app WITHOUT wiping its data, so the
# sideloaded Gemma model (see tool/push-model.sh) survives every iteration and
# is never re-downloaded.
#
#   bash tool/dev-install.sh                 # pick a built/downloaded APK, install -r, launch
#   bash tool/dev-install.sh path/to.apk     # install a specific APK
#   bash tool/dev-install.sh --with-model     # also ensure the model is on the device
#   bash tool/dev-install.sh -y               # don't prompt before a clean reinstall
#
# With no APK argument it uses, in order: a locally built debug APK, a staged
# ./speaker-android.apk, else it downloads the latest CI release APK into
# .dev-cache/. `adb install -r` keeps app data; if signatures don't match
# (e.g. a locally built APK over a CI one) it offers a clean reinstall and then
# re-pushes the model for you.
set -euo pipefail

cd "$(dirname "$0")/.."

APP_ID="${SPEAKER_APP_ID:-com.k2fsa.sherpa.speaker_mobile}"
REPO="${SPEAKER_REPO:-DobosP/speaker}"
APK_URL="${SPEAKER_APK_URL:-https://github.com/${REPO}/releases/download/android-latest/speaker-android.apk}"
ADB="${ADB:-adb}"
ASSUME_YES=0
WITH_MODEL=0
APK=""

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    --with-model) WITH_MODEL=1; shift ;;
    --app-id) APP_ID="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) APK="$1"; shift ;;
  esac
done

command -v "$ADB" >/dev/null 2>&1 || { echo "ERROR: adb not found (set \$ADB or add it to PATH)." >&2; exit 1; }

confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  if [ ! -t 0 ]; then echo "(non-interactive; re-run with -y to allow)" >&2; return 1; fi
  read -r -p "$1 [y/N] " ans
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# Resolve which APK to install.
if [ -z "$APK" ]; then
  for cand in \
    build/app/outputs/flutter-apk/app-debug.apk \
    speaker-android.apk \
    .dev-cache/speaker-android.apk; do
    if [ -f "$cand" ]; then APK="$cand"; break; fi
  done
fi
if [ -z "$APK" ]; then
  echo "==> No local APK found; downloading the latest release APK"
  mkdir -p .dev-cache
  curl -fL --retry 3 --progress-bar -o .dev-cache/speaker-android.apk "$APK_URL"
  APK=".dev-cache/speaker-android.apk"
fi
[ -f "$APK" ] || { echo "ERROR: APK not found: $APK" >&2; exit 1; }
echo "==> Installing $APK (keeping app data)"

# install -r keeps data; -d allows a version-code downgrade between builds.
out="$("$ADB" install -r -d "$APK" 2>&1)" && rc=0 || rc=$?
echo "$out"

if [ "$rc" != "0" ]; then
  if echo "$out" | grep -qiE 'INSTALL_FAILED_UPDATE_INCOMPATIBLE|signatures do not match|INSTALL_FAILED_VERSION_DOWNGRADE'; then
    echo
    echo "Signature/version mismatch: this APK can't replace the installed one in place."
    echo "A clean reinstall fixes it, but it WIPES app data — including the on-device"
    echo "model, which this script will re-push afterwards."
    if confirm "Uninstall $APP_ID and install fresh?"; then
      "$ADB" uninstall "$APP_ID" >/dev/null 2>&1 || true
      "$ADB" install -d "$APK"
      echo "==> Clean install done; restoring the model"
      bash tool/push-model.sh --app-id "$APP_ID" || \
        echo "WARNING: model re-push failed; run tool/push-model.sh manually."
    else
      echo "Aborted. (Old build left installed.)"
      exit 1
    fi
  else
    echo "ERROR: install failed (see output above)." >&2
    exit 1
  fi
fi

if [ "$WITH_MODEL" = "1" ]; then
  echo "==> Ensuring the model is on the device"
  bash tool/push-model.sh --app-id "$APP_ID"
fi

# Launch without needing the exact activity name.
"$ADB" shell monkey -p "$APP_ID" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
echo "Done. App (re)installed and launched. The sideloaded model is reused — no re-download."
