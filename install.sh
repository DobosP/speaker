#!/usr/bin/env bash
# One-command native install for the local voice assistant (Linux/macOS).
#
#   ./install.sh                 full install (system audio lib, venv, deps, models)
#   ./install.sh --skip-system   skip the apt/brew PortAudio step (no sudo)
#   ./install.sh --recreate      rebuild the venv from scratch (fixes conda/venv mixes)
#   ./install.sh --dry-run       print the plan, change nothing
#   ./install.sh --obsidian-vault PATH --enable-reminders
#                                configure optional assistant capabilities
#
# Thin wrapper: it does the OS-specific PortAudio step, then hands the
# cross-platform work (venv with pip, deps, models, doctor) to tools/install.py
# so Linux/macOS/Windows share one code path. Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
SKIP_SYSTEM=0
FORWARD=()
for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    *) FORWARD+=("$arg") ;;   # all cross-platform installer/setup options
  esac
done

echo "==> System audio library (PortAudio)"
if [ "$SKIP_SYSTEM" = 1 ]; then
  echo "    skipped (--skip-system)"
elif [ "$(uname)" = "Darwin" ]; then
  if command -v brew >/dev/null 2>&1 && ! brew list portaudio >/dev/null 2>&1; then
    echo "    installing portaudio via Homebrew"
    brew install portaudio || echo "    (brew install failed; the sounddevice wheel usually bundles PortAudio)"
  else
    echo "    portaudio present or Homebrew unavailable (the wheel usually bundles it)"
  fi
elif command -v apt-get >/dev/null 2>&1; then
  if dpkg -s libportaudio2 >/dev/null 2>&1; then
    echo "    libportaudio2 already installed"
  else
    echo "    installing libportaudio2 portaudio19-dev (needs sudo)"
    sudo apt-get update -qq && sudo apt-get install -y libportaudio2 portaudio19-dev
  fi
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y portaudio || true
elif command -v pacman >/dev/null 2>&1; then
  sudo pacman -S --noconfirm portaudio || true
else
  echo "    no known package manager -- install PortAudio manually if mic/speaker fail"
fi

echo "==> Cross-platform install (venv, deps, models, doctor)"
exec "$PY" tools/install.py "${FORWARD[@]}"
