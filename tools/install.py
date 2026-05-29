#!/usr/bin/env python3
"""Cross-platform one-command installer for the native voice runtime.

One code path for Linux, Windows, and macOS. The OS-specific wrappers
(``install.sh`` / ``install.ps1`` / ``install.bat``) just locate a system
Python and hand off to this module, so the real logic lives in one place and is
unit-testable.

    python tools/install.py            # create .venv, install deps, fetch models, doctor
    python tools/install.py --dry-run  # print the plan, change nothing
    python tools/install.py --help

What it does (idempotent -- safe to re-run):
  1. Create (or reuse) a clean ``.venv`` that actually has pip -- this is the
     fix for the broken conda/venv mixes that fail with "No module named pip".
  2. Install the lean runtime deps (no torch). On Windows the sounddevice wheel
     bundles PortAudio, so there is no system step; on Linux PortAudio is a
     system package (the install.sh wrapper handles the sudo apt step).
  3. Download the speech models + wire config.local.json (tools.setup_models).
  4. Run the preflight doctor and print OS-correct next steps.

The LLM (Ollama) is separate -- see the printed notes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# The lean on-device runtime. Deliberately no torch: sherpa-onnx is the whole
# STT/VAD/TTS/speaker-ID stack, Ollama serves the LLM out of process. Keep this
# list in sync with install.sh's comment and README.
RUNTIME_DEPS = [
    "sounddevice",   # mic/speaker I/O (bundles PortAudio on Windows/macOS wheels)
    "sherpa-onnx",   # STT + VAD + TTS + speaker embeddings (ONNX, CPU)
    "numpy",
    "ollama",        # local LLM client
    "huggingface-hub",  # model downloads
    "psutil",        # CPU/RAM telemetry in the run summary
]

DEFAULT_VENV = ".venv"


# --- pure, OS-aware helpers (unit-tested for both Windows and POSIX) ---------


def is_windows(system: str | None = None) -> bool:
    """True on Windows. ``system`` overrides the detected platform for tests."""
    plat = system if system is not None else sys.platform
    return plat.startswith("win")


def venv_python_path(venv_dir: str, *, system: str | None = None) -> str:
    """Path to the Python interpreter inside ``venv_dir``.

    Windows puts it in ``Scripts\\python.exe``; POSIX in ``bin/python``. Returned
    with the right separators for the target OS so the wrappers and the doctor
    can print a path the user can actually run."""
    if is_windows(system):
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def activation_hint(venv_dir: str, *, system: str | None = None) -> str:
    """The shell command that activates ``venv_dir`` on the target OS."""
    if is_windows(system):
        # PowerShell is the default Windows shell; cmd users use the .bat form.
        return f"{venv_dir}\\Scripts\\Activate.ps1   (cmd: {venv_dir}\\Scripts\\activate.bat)"
    return f"source {venv_dir}/bin/activate"


def portaudio_hint(*, system: str | None = None) -> str:
    """How to get PortAudio (sounddevice's native dep) on the target OS."""
    plat = system if system is not None else sys.platform
    if plat.startswith("win"):
        return "PortAudio ships inside the sounddevice wheel on Windows -- nothing to install."
    if plat == "darwin":
        return "macOS: `brew install portaudio` if mic/speaker fail (the wheel usually bundles it)."
    return (
        "Linux: install PortAudio if mic/speaker fail -- "
        "Debian/Ubuntu: `sudo apt-get install libportaudio2 portaudio19-dev`; "
        "Fedora: `sudo dnf install portaudio`; Arch: `sudo pacman -S portaudio`."
    )


def running_in_conda() -> bool:
    """True if a conda environment is active.

    The reported failure (`No module named pip` in `.venv`) came from a venv
    created by a conda-base Python; we warn so the user knows to build the venv
    from a clean interpreter (which `--recreate` does)."""
    return bool(os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX"))


def needs_fresh_venv(
    venv_dir: str,
    *,
    exists: "callable" = os.path.exists,
    pip_ok: "callable | None" = None,
) -> bool:
    """Whether to (re)create the venv.

    True when the venv interpreter is missing, or it exists but its pip is
    broken (``pip_ok`` returns False) -- the exact conda-mix breakage. ``exists``
    and ``pip_ok`` are injectable so the decision logic is testable without a
    real filesystem or subprocess."""
    py = venv_python_path(venv_dir)
    if not exists(py):
        return True
    if pip_ok is None:
        return False
    return not pip_ok(py)


def install_plan(args, *, system: str | None = None) -> list[str]:
    """Human-readable list of the steps that *would* run -- powers --dry-run and
    keeps the orchestration and its description from drifting apart."""
    venv = args.venv
    py = venv_python_path(venv, system=system)
    steps = [
        f"create/refresh venv at {venv} (recreate={args.recreate})",
        f"install runtime deps into {py}: {', '.join(RUNTIME_DEPS)}",
    ]
    if not args.skip_models:
        steps.append(f"{py} -m tools.setup_models  (download speech models + wire config.local.json)")
    else:
        steps.append("skip model download (--skip-models)")
    steps.append(f"{py} -m tools.doctor  (preflight check)")
    return steps


# --- orchestration (subprocess-driven; exercised via --dry-run) --------------


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _pip_works(py: str) -> bool:
    try:
        return subprocess.call(
            [py, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) == 0
    except OSError:
        return False


def create_venv(venv_dir: str, *, base_python: str, recreate: bool, dry_run: bool) -> str:
    """Create ``venv_dir`` with a working pip; return its interpreter path.

    Rebuilds when the venv is missing or its pip is broken (or ``recreate``).
    Uses ``ensurepip`` as a fallback so a distro that ships venv without pip
    (Debian's ``python3-venv`` gap) still ends up usable."""
    py = venv_python_path(venv_dir)
    fresh = recreate or needs_fresh_venv(venv_dir, pip_ok=_pip_works)
    if not fresh:
        print(f"    reusing existing venv at {venv_dir}")
        return py
    if recreate and os.path.exists(venv_dir) and not dry_run:
        import shutil

        shutil.rmtree(venv_dir, ignore_errors=True)
    print(f"    creating venv at {venv_dir}")
    # --clear wipes any half-built/conda-polluted venv so we start clean.
    if _run([base_python, "-m", "venv", "--clear", venv_dir], dry_run=dry_run) != 0:
        raise SystemExit(
            "venv creation failed. On Debian/Ubuntu: `sudo apt-get install python3-venv`."
        )
    # Ensure pip exists inside the venv (the conda-mix fix), then upgrade it.
    _run([py, "-m", "ensurepip", "--upgrade"], dry_run=dry_run)
    _run([py, "-m", "pip", "install", "--upgrade", "pip"], dry_run=dry_run)
    return py


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform installer for the native voice runtime",
    )
    parser.add_argument("--venv", default=DEFAULT_VENV, help=f"venv dir (default: {DEFAULT_VENV})")
    parser.add_argument(
        "--python", dest="base_python", default=sys.executable,
        help="base Python used to build the venv (default: the one running this script)",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="delete and rebuild the venv from scratch (use after a conda/venv mix)",
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="don't download speech models (deps + venv only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and the exact commands, but change nothing",
    )
    args = parser.parse_args(argv)

    print("== Voice assistant installer ==")
    print(f"   OS: {sys.platform}   base python: {args.base_python}")
    if running_in_conda():
        print("   note: a conda env is active. Building an isolated .venv from it is fine, "
              "but if pip looks broken later, re-run with --recreate.")
    print(f"   {portaudio_hint()}")
    print()

    if args.dry_run:
        print("Plan (--dry-run, nothing will change):")
        for i, step in enumerate(install_plan(args), 1):
            print(f"  {i}. {step}")
        print("\nRun without --dry-run to execute.")
        return 0

    print("==> 1/4 Python virtual environment")
    py = create_venv(args.venv, base_python=args.base_python,
                     recreate=args.recreate, dry_run=False)

    print("==> 2/4 Runtime dependencies (lean, no torch)")
    if _run([py, "-m", "pip", "install", *RUNTIME_DEPS], dry_run=False) != 0:
        print("    dependency install failed -- see the pip output above.", file=sys.stderr)
        return 1

    if not args.skip_models:
        print("==> 3/4 Speech models + config wiring")
        # Run setup_models from the repo root so relative paths resolve; it is
        # non-fatal here because the doctor will report exactly what's missing.
        _run([py, "-m", "tools.setup_models"], dry_run=False)
    else:
        print("==> 3/4 Speech models (skipped: --skip-models)")

    print("==> 4/4 Preflight check")
    _run([py, "-m", "tools.doctor"], dry_run=False)

    print("\nDone. To run the assistant:\n")
    print(f"    {activation_hint(args.venv)}")
    print("    python -m core --engine sherpa\n")
    print("Try it with no audio/models first:")
    print("    python -m core --engine console --llm echo\n")
    print("The LLM runs in Ollama (separate; https://ollama.com). Then:")
    print("    ollama pull gemma3:12b && ollama pull gemma3:4b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
