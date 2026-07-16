#!/usr/bin/env bash
# One-command private Linux live session. The Python launcher owns transient
# Ollama/PipeWire setup, readiness, recording, and exact-resource cleanup.
set -euo pipefail
cd "$(dirname "$0")"

PY="${SPEAKER_PYTHON:-python3}"
if [ -z "${SPEAKER_PYTHON:-}" ] && [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
fi

exec "$PY" -m tools.live_launcher "$@"
