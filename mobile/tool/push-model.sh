#!/usr/bin/env bash
# Put the on-device Gemma weights on the phone ONCE, so the app never downloads
# them over the network again. Pushes to the app's stable model path
# (ModelStore in lib/model_store.dart), which survives `adb install -r`, so the
# normal test loop becomes: push the model once, then reinstall the APK freely.
#
#   bash tool/push-model.sh                 # download to .dev-cache, then push
#   bash tool/push-model.sh --model a.task  # push an already-local file
#   SPEAKER_APP_ID=... bash tool/push-model.sh
#
# Re-run this after a *clean* install (uninstall+install wipes app storage);
# tool/dev-install.sh does that for you automatically.
set -euo pipefail

cd "$(dirname "$0")/.."

APP_ID="${SPEAKER_APP_ID:-com.k2fsa.sherpa.speaker_mobile}"
REPO="${SPEAKER_REPO:-DobosP/speaker}"
MODEL_FILE="Gemma3-1B-IT-q4.task"
MODEL_URL="${SPEAKER_MODEL_URL:-https://github.com/${REPO}/releases/download/gemma-model/${MODEL_FILE}}"
MIN_BYTES=$((100 * 1024 * 1024)) # sanity floor; the real file is ~550 MB
ADB="${ADB:-adb}"

CACHE_DIR=".dev-cache"
LOCAL="$CACHE_DIR/$MODEL_FILE"

# Where lib/model_store.dart reads the model from on Android: the app's external
# files dir + /models. (/sdcard is /storage/emulated/0.)
DEST_DIR="/sdcard/Android/data/${APP_ID}/files/models"
DEST="$DEST_DIR/$MODEL_FILE"

while [ $# -gt 0 ]; do
  case "$1" in
    --model) LOCAL="$2"; shift 2 ;;
    --app-id) APP_ID="$2"; DEST_DIR="/sdcard/Android/data/${APP_ID}/files/models"; DEST="$DEST_DIR/$MODEL_FILE"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v "$ADB" >/dev/null 2>&1 || { echo "ERROR: adb not found (set \$ADB or add it to PATH)." >&2; exit 1; }

filesize() { wc -c < "$1" | tr -d ' '; }

# 1) Make sure we have a complete local copy of the weights.
if [ -f "$LOCAL" ] && [ "$(filesize "$LOCAL")" -ge "$MIN_BYTES" ]; then
  echo "==> Using cached model: $LOCAL ($(filesize "$LOCAL") bytes)"
else
  echo "==> Downloading model to $LOCAL"
  mkdir -p "$(dirname "$LOCAL")"
  curl -fL --retry 3 --progress-bar -o "$LOCAL.part" "$MODEL_URL"
  if [ "$(filesize "$LOCAL.part")" -lt "$MIN_BYTES" ]; then
    echo "ERROR: download looks truncated ($(filesize "$LOCAL.part") bytes < $MIN_BYTES)." >&2
    rm -f "$LOCAL.part"; exit 1
  fi
  mv -f "$LOCAL.part" "$LOCAL"
  echo "==> Downloaded $(filesize "$LOCAL") bytes"
fi
LOCAL_SIZE="$(filesize "$LOCAL")"

# 2) The app's external dir only exists once the app is installed.
if ! "$ADB" shell pm path "$APP_ID" >/dev/null 2>&1; then
  echo "ERROR: package '$APP_ID' is not installed on the device." >&2
  echo "       - Install it first:  bash tool/dev-install.sh" >&2
  echo "       - Or find the real id:  $ADB shell pm list packages | grep -i sherpa" >&2
  echo "         then re-run with:  SPEAKER_APP_ID=<id> bash tool/push-model.sh" >&2
  exit 1
fi

# 3) Push. Prefer a direct adb push; fall back to streaming through run-as for
#    devices that block shell access to /Android/data (works on debug builds).
echo "==> Pushing model to $DEST"
"$ADB" shell mkdir -p "$DEST_DIR" >/dev/null 2>&1 || true
if "$ADB" push "$LOCAL" "$DEST" >/dev/null 2>&1; then
  echo "    pushed directly"
else
  echo "    direct push blocked — falling back to run-as (debug build)"
  if ! "$ADB" shell run-as "$APP_ID" sh -c "mkdir -p '$DEST_DIR' && cat > '$DEST'" < "$LOCAL"; then
    echo "ERROR: could not write the model to the device." >&2
    echo "       Is this a debuggable build, and is USB debugging authorized?" >&2
    exit 1
  fi
  echo "    streamed via run-as"
fi

# 4) Verify the on-device size matches (run-as reads it even if shell can't).
remote_size="$("$ADB" shell run-as "$APP_ID" sh -c "wc -c < '$DEST'" 2>/dev/null | tr -dc '0-9' || true)"
[ -n "$remote_size" ] || remote_size="$("$ADB" shell sh -c "wc -c < '$DEST'" 2>/dev/null | tr -dc '0-9' || true)"
if [ -z "$remote_size" ]; then
  echo "WARNING: pushed, but could not read back the on-device size to verify."
elif [ "$remote_size" != "$LOCAL_SIZE" ]; then
  echo "ERROR: on-device size ($remote_size) != local size ($LOCAL_SIZE). Push corrupted." >&2
  exit 1
else
  echo "==> Verified: $remote_size bytes on device."
fi

echo "Done. The app will load this file instead of downloading. Path:"
echo "  $DEST"
