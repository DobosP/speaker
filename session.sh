#!/usr/bin/env bash
# Capture a full debuggable session in one go: verbose console + audio recording
# + logs + transcript + per-turn timings + CPU/GPU/RAM telemetry, all bundled
# under one run id in logs/runs/run-<id>.{txt,summary.json,wav}.
#
#   ./session.sh                 # records a sherpa (mic) session
#   ENGINE=console ./session.sh  # text session (no audio to record)
#   ./session.sh --llm echo      # extra args pass straight through
#
# When it exits (clean, Ctrl-C, or crash) the bundle is written. Commit + push
# all three files so they can be replayed and debugged.
set -euo pipefail
cd "$(dirname "$0")"

PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python

"$PY" -m core --engine "${ENGINE:-sherpa}" --debug --record "$@"

echo
echo "Session bundle written under logs/runs/ (run-<id>.txt / .summary.json / .wav)."
echo "Commit + push it for debugging:"
echo "    git add logs/runs/run-*.txt logs/runs/run-*.summary.json logs/runs/run-*.wav"
echo "    git commit -m 'session capture' && git push"
