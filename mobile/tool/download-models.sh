#!/usr/bin/env bash
# Download the on-device ASR + TTS models into ./assets/.
# Runs at build time (CI or local) so large model binaries are never committed.
set -euo pipefail

WITH_WHISPER=false
case "${1:-}" in
  "") ;;
  --with-whisper) WITH_WHISPER=true; shift ;;
  --help|-h)
    echo "Usage: $0 [--with-whisper]"
    echo "Default: streaming Zipformer + Piper. Optional Whisper is not used by the live assistant."
    echo "To bundle Whisper too, also run generate-asset-list.py --with-whisper."
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--with-whisper]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
mkdir -p assets
cd assets

ASR=sherpa-onnx-streaming-zipformer-en-2023-06-26
# Optional offline recognizer retained for explicit future/evidence use.
# The shipped assistant has no second-pass call.
WHISPER=sherpa-onnx-whisper-base.en
TTS=vits-piper-en_US-amy-low
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download

fetch() {
  local name="$1" url="$2"
  if [ -d "$name" ]; then
    echo "==> $name already present, skipping"
    return
  fi
  echo "==> downloading $name"
  curl -sSL "$url" -o "$name.tar.bz2"
  tar xf "$name.tar.bz2"
  rm -f "$name.tar.bz2"
}

fetch "$ASR" "$BASE/asr-models/$ASR.tar.bz2"
if [ "$WITH_WHISPER" = true ]; then
  fetch "$WHISPER" "$BASE/asr-models/$WHISPER.tar.bz2"
fi
fetch "$TTS" "$BASE/tts-models/$TTS.tar.bz2"

# Bundle filtering happens in generate-asset-list.py. Preserve cached optional
# weights and examples on disk even when they are excluded from the APK.

echo "==> models ready:"
du -sh "$ASR" "$TTS"
if [ "$WITH_WHISPER" = true ]; then
  du -sh "$WHISPER"
fi
