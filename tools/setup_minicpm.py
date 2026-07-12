"""Provision the supported MiniCPM5-1B Ollama answering model.

The official GGUF is public, but a direct ``ollama pull hf.co/...`` gets an
auto-generated template whose stop set is not reliable with this project's
system prompt.  This helper pulls the official Q8 artifact and creates the
stable local alias referenced by ``config.json`` using the committed OpenBMB
ChatML Modelfile.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from core.minicpm_identity import (
    MINICPM_Q8_CONTRACT,
    ModelIdentity,
    validate_minicpm_modelfile,
    verify_minicpm_q8_identity,
)


# Historical public names remain imports for evaluator/tests, but their source
# of truth is the production identity contract above.
SOURCE_MODEL = MINICPM_Q8_CONTRACT.source
SOURCE_MODEL_BLOB_SHA256 = MINICPM_Q8_CONTRACT.blob_sha256
LOCAL_MODEL = MINICPM_Q8_CONTRACT.alias
DEFAULT_MODELFILE = MINICPM_Q8_CONTRACT.modelfile


class MiniCPMIdentityError(RuntimeError):
    """Provisioning ran, but the supported alias identity was not proven."""

    def __init__(self, message: str, *, identity: ModelIdentity | None = None):
        super().__init__(message)
        self.identity = identity


def _validate_requested_identity(
    *, source_model: str, local_model: str, modelfile: Path
) -> None:
    if source_model != SOURCE_MODEL:
        raise MiniCPMIdentityError(
            f"unsupported MiniCPM source {source_model!r}; expected {SOURCE_MODEL}"
        )
    if local_model != LOCAL_MODEL:
        raise MiniCPMIdentityError(
            f"unsupported MiniCPM alias {local_model!r}; expected {LOCAL_MODEL}"
        )
    if modelfile.resolve() != DEFAULT_MODELFILE.resolve():
        raise MiniCPMIdentityError(
            f"unsupported MiniCPM Modelfile {modelfile}; expected {DEFAULT_MODELFILE}"
        )


def _default_show(
    *, client_factory: Callable[..., object] | None = None
) -> Callable[[str], object]:
    if client_factory is None:
        import ollama

        client_factory = ollama.Client

    # Ollama's CLI and SDK both honor OLLAMA_HOST.  Using the same process
    # environment binds post-create inspection to the daemon just provisioned.
    kwargs: dict[str, object] = {"timeout": 5.0}
    host = str(os.environ.get("OLLAMA_HOST", "") or "").strip()
    if host:
        kwargs["host"] = host
    client = client_factory(**kwargs)
    return client.show


def verify_installed(
    *, show: Callable[[str], object] | None = None
) -> ModelIdentity:
    """Verify the installed canonical alias without pulling or creating."""
    try:
        inspector = show or _default_show()
    except Exception as exc:  # noqa: BLE001 - controlled setup failure
        raise MiniCPMIdentityError(
            f"identity inspector unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    identity = verify_minicpm_q8_identity(show=inspector)
    if not identity.ok:
        raise MiniCPMIdentityError(
            identity.error or "MiniCPM identity verification failed",
            identity=identity,
        )
    return identity


def provision(
    *,
    source_model: str = SOURCE_MODEL,
    local_model: str = LOCAL_MODEL,
    modelfile: Path = DEFAULT_MODELFILE,
    pull: bool = True,
    runner: Callable[..., object] = subprocess.run,
    show: Callable[[str], object] | None = None,
) -> ModelIdentity:
    """Create and then prove the one supported desktop Ollama identity."""
    _validate_requested_identity(
        source_model=source_model,
        local_model=local_model,
        modelfile=modelfile,
    )
    validate_minicpm_modelfile(modelfile)
    commands: list[Sequence[str]] = []
    if pull:
        commands.append(("ollama", "pull", source_model))
    commands.append(("ollama", "create", local_model, "-f", str(modelfile)))
    for command in commands:
        runner(list(command), check=True)
    return verify_installed(show=show)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=SOURCE_MODEL)
    parser.add_argument("--model", default=LOCAL_MODEL, help="local Ollama alias")
    parser.add_argument("--modelfile", type=Path, default=DEFAULT_MODELFILE)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="inspect the canonical alias/source without pulling or creating",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="reuse an already-pulled source model and only recreate the alias",
    )
    args = parser.parse_args(argv)
    try:
        _validate_requested_identity(
            source_model=args.source_model,
            local_model=args.model,
            modelfile=args.modelfile,
        )
        if args.verify_only:
            identity = verify_installed()
        else:
            identity = provision(
                source_model=args.source_model,
                local_model=args.model,
                modelfile=args.modelfile,
                pull=not args.no_pull,
            )
    except (
        FileNotFoundError,
        ImportError,
        MiniCPMIdentityError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"MiniCPM setup failed: {exc}\n")
    action = "identity verified" if args.verify_only else "ready"
    print(
        f"MiniCPM {action} as {identity.alias} "
        f"({identity.quantization}, sha256:{identity.alias_blob})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
