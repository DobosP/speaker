#!/usr/bin/env bash
# Capture a full debuggable session in one go: verbose console + audio recording
# + logs + transcript + per-turn timings + CPU/GPU/RAM telemetry, all bundled
# under one run id in logs/runs/run-<id>.{txt,summary.json,wav}.
#
#   ./session.sh                 # low-level sherpa capture (prerequisites ready)
#   ENGINE=console ./session.sh  # text session (no audio to record)
#   ./session.sh --llm echo      # extra args pass straight through
#
# On Linux, prefer ./live.sh for conditional Ollama plus reversible PipeWire
# setup and a private aligned mic/reference capture. Real-voice bundles stay local.
set -euo pipefail
cd "$(dirname "$0")"

PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python

"$PY" -m core --engine "${ENGINE:-sherpa}" --debug --record "$@"

echo
echo "Session finalized. Use the [log] paths above for local replay/debugging."
echo "Keep real voice, transcripts, and prompts local; do not commit or push them."
