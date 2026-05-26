#!/usr/bin/env bash
# Live TTS diagnostic run — tee stderr/stdout to a log file for grep analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${SPEAKER_TTS_LOG:-/tmp/speaker-tts-debug.log}"
export SPEAKER_TTS_DEBUG=1
exec python main.py "$@" 2>&1 | tee "$LOG"
