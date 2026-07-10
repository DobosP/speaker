"""Provision the supported MiniCPM5-1B Ollama answering model.

The official GGUF is public, but a direct ``ollama pull hf.co/...`` gets an
auto-generated template whose stop set is not reliable with this project's
system prompt.  This helper pulls the official Q8 artifact and creates the
stable local alias referenced by ``config.json`` using the committed OpenBMB
ChatML Modelfile.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Callable, Sequence

SOURCE_MODEL = "hf.co/openbmb/MiniCPM5-1B-GGUF:Q8_0"
LOCAL_MODEL = "minicpm5-1b:q8"
DEFAULT_MODELFILE = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "ollama"
    / "Modelfile.minicpm5-1b-q8"
)


def provision(
    *,
    source_model: str = SOURCE_MODEL,
    local_model: str = LOCAL_MODEL,
    modelfile: Path = DEFAULT_MODELFILE,
    pull: bool = True,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Pull the official GGUF and create the configured local Ollama alias."""
    if not modelfile.is_file():
        raise FileNotFoundError(f"MiniCPM Modelfile not found: {modelfile}")
    commands: list[Sequence[str]] = []
    if pull:
        commands.append(("ollama", "pull", source_model))
    commands.append(("ollama", "create", local_model, "-f", str(modelfile)))
    for command in commands:
        runner(list(command), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=SOURCE_MODEL)
    parser.add_argument("--model", default=LOCAL_MODEL, help="local Ollama alias")
    parser.add_argument("--modelfile", type=Path, default=DEFAULT_MODELFILE)
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="reuse an already-pulled source model and only recreate the alias",
    )
    args = parser.parse_args(argv)
    try:
        provision(
            source_model=args.source_model,
            local_model=args.model,
            modelfile=args.modelfile,
            pull=not args.no_pull,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"MiniCPM setup failed: {exc}\n")
    print(f"MiniCPM ready as {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
