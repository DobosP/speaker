#!/usr/bin/env bash
# Round-2 capture-calibration driver (Linux/PipeWire).
# Session-2 helper for tools/calibration_suite.py — NOT committed logic, just a
# convenience wrapper so the voice_comm* presets record through the PipeWire
# echo-cancel node (the tool's capture_voice_comm flag is a WASAPI-only no-op on
# Linux; the OS voice-comm path here == recording via module-echo-cancel's source).
#
# Usage:
#   tools/calib_round2.sh inapp   <out_folder>   # raw,denoise,apm via the real mic
#   tools/calib_round2.sh vc      <out_folder>   # voice_comm,voice_comm_denoise via EC source
#   tools/calib_round2.sh vc-talk <out_folder>   # double-talk: TTS through EC sink, record EC source
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODE="${1:?mode: inapp|vc|vc-talk}"
OUT="${2:?output folder}"
SECS="${SECS:-12}"
EC_SRC=ec_source
EC_SINK=ec_sink

restore() { pactl set-default-source "$ORIG_SRC" >/dev/null 2>&1 || true
            pactl set-default-sink   "$ORIG_SINK" >/dev/null 2>&1 || true; }

case "$MODE" in
  inapp)
    $PY -m tools.calibration_suite --presets raw,denoise,apm \
        --play --seconds "$SECS" --out "$OUT"
    ;;
  vc)
    ORIG_SRC=$(pactl get-default-source); ORIG_SINK=$(pactl get-default-sink)
    trap restore EXIT
    pactl set-default-source "$EC_SRC"
    echo "[round2] default source -> $EC_SRC (OS voice-comm path)"
    $PY -m tools.calibration_suite --presets voice_comm,voice_comm_denoise \
        --play --seconds "$SECS" --out "$OUT"
    ;;
  vc-talk)
    ORIG_SRC=$(pactl get-default-source); ORIG_SINK=$(pactl get-default-sink)
    trap restore EXIT
    pactl set-default-source "$EC_SRC"; pactl set-default-sink "$EC_SINK"
    echo "[round2] source -> $EC_SRC, sink -> $EC_SINK (echo ref aligned for double-talk)"
    $PY -m tools.calibration_suite --presets voice_comm,voice_comm_denoise \
        --talk-over --seconds "$SECS" --out "$OUT"
    ;;
  *) echo "unknown mode: $MODE"; exit 2;;
esac
