"""Ollama identity adapters for the conversation evaluation harness."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Callable, Mapping

from core.minicpm_identity import (
    ModelIdentity,
    _blob_ref,
    _effective_config_digest,
    _expected_template,
    verify_minicpm_q8_identity,
)


_NO_AMBIENT_AUTH = {"authorization": "Bearer speaker-local-evaluation"}


@dataclass(frozen=True)
class OllamaBlobIdentity:
    model: str
    blob_sha256: str
    effective_config_sha256: str
    ok: bool
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _ollama_show(
    *,
    host: str | None,
    client_headers: Mapping[str, str] | None,
    timeout_sec: float,
) -> Callable[[str], object]:
    import ollama

    client = ollama.Client(
        host=host,
        headers=client_headers or _NO_AMBIENT_AUTH,
        timeout=max(0.1, float(timeout_sec)),
    )
    return client.show


def verify_minicpm_identity(
    *,
    show: Callable[[str], object] | None = None,
    host: str | None = None,
    client_headers: Mapping[str, str] | None = None,
    timeout_sec: float = 5.0,
) -> ModelIdentity:
    """Use the production MiniCPM contract with evaluator transport policy."""
    if show is None:
        try:
            show = _ollama_show(
                host=host,
                client_headers=client_headers,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 - identity evidence fails closed
            def failed_show(_model: str, error: Exception = exc) -> object:
                raise error

            show = failed_show
    return verify_minicpm_q8_identity(show=show)


def verify_ollama_blob_identity(
    model: str,
    *,
    show: Callable[[str], object] | None = None,
    host: str | None = None,
    client_headers: Mapping[str, str] | None = None,
    timeout_sec: float = 5.0,
) -> OllamaBlobIdentity:
    """Pin a generic Ollama alias's immutable blob and effective config."""
    try:
        if show is None:
            show = _ollama_show(
                host=host,
                client_headers=client_headers,
                timeout_sec=timeout_sec,
            )
        shown = show(model)
        blob = _blob_ref(shown)
        config_digest = _effective_config_digest(shown)
        ok = bool(
            re.fullmatch(r"[0-9a-f]{64}", blob, re.IGNORECASE) is not None
            and re.fullmatch(
                r"[0-9a-f]{64}", config_digest, re.IGNORECASE
            )
            is not None
        )
        return OllamaBlobIdentity(
            model=model,
            blob_sha256=blob,
            effective_config_sha256=config_digest,
            ok=ok,
        )
    except Exception as exc:
        return OllamaBlobIdentity(
            model=model,
            blob_sha256="",
            effective_config_sha256="",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
