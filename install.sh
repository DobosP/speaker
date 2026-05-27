#!/usr/bin/env bash
# One-command native install for the local voice assistant (Linux).
#
#   ./install.sh                 full install (system audio lib, venv, deps, models)
#   ./install.sh --skip-system   skip the apt PortAudio step (no sudo)
#
# Creates a clean .venv WITH pip (sidestepping broken conda/venv mixes),
# installs the lean runtime deps, downloads the speech models + wires
# config.json, then runs the preflight doctor. The LLM (Ollama) is separate --
# see the final notes. Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
SKIP_SYSTEM=0
for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

echo "==> 1/5 System audio library (PortAudio)"
if [ "$SKIP_SYSTEM" = 1 ]; then
  echo "    skipped (--skip-system)"
elif command -v apt-get >/dev/null 2>&1; then
  if dpkg -s libportaudio2 >/dev/null 2>&1; then
    echo "    libportaudio2 already installed"
  else
    echo "    installing libportaudio2 portaudio19-dev (needs sudo)"
    sudo apt-get update -qq && sudo apt-get install -y libportaudio2 portaudio19-dev
  fi
else
  echo "    no apt-get found -- install PortAudio with your package manager if mic/speaker fail"
fi

echo "==> 2/5 Python virtual environment (.venv)"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  echo "    creating a fresh .venv with pip"
  rm -rf .venv
  "$PY" -m venv .venv || {
    echo "    venv creation failed -- on Debian/Ubuntu: sudo apt-get install python3-venv"
    exit 1
  }
  .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
else
  echo "    reusing existing .venv"
fi
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip >/dev/null

echo "==> 3/5 Python dependencies (lean runtime, no torch)"
# psutil powers the CPU/RAM telemetry in the run summary (GPU uses nvidia-smi).
"$VPY" -m pip install sounddevice sherpa-onnx numpy ollama huggingface-hub psutil

echo "==> 4/5 Speech models + config wiring"
"$VPY" -m tools.setup_models

echo "==> 5/5 Preflight check"
"$VPY" -m tools.doctor || true

cat <<'EOF'

Done. To run the assistant:

    source .venv/bin/activate
    python -m core --engine sherpa

The LLM runs in Ollama (separate, uses your GPU). If it is not installed yet:
    https://ollama.com
Then pull the models the desktop profile uses:
    ollama pull gemma3:12b
    ollama pull gemma3:4b

If `python -m tools.doctor` still shows FAIL lines, fix those first.
EOF
