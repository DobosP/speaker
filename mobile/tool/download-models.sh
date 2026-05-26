#!/usr/bin/env bash
# Download the on-device ASR + TTS models into ./assets/.
# Runs at build time (CI or local) so large model binaries are never committed.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p assets
cd assets

ASR=sherpa-onnx-streaming-zipformer-en-2023-06-26
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
fetch "$TTS" "$BASE/tts-models/$TTS.tar.bz2"

echo "==> models ready:"
du -sh "$ASR" "$TTS"
