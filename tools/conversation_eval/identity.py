from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable, Mapping

from tools.setup_minicpm import (
    DEFAULT_MODELFILE,
    LOCAL_MODEL,
    SOURCE_MODEL,
    SOURCE_MODEL_BLOB_SHA256,
)


_MODELFILE_TEMPLATE_RE = re.compile(
    r"^TEMPLATE\s+(.*?)(?=^PARAMETER\s|\Z)",
    re.DOTALL | re.MULTILINE,
)
_FROM_RE = re.compile(r"^FROM\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_DIGEST_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)
_NO_AMBIENT_AUTH = {"authorization": "Bearer speaker-local-evaluation"}


@dataclass(frozen=True)
class ModelIdentity:
    alias: str
    source: str
    alias_blob_sha256: str
    source_blob_sha256: str
    alias_blob: str
    source_blob: str
    blob_match: bool
    pinned_blob_match: bool
    quantization: str
    q8: bool
    template_match: bool
    parameters_match: bool
    effective_config_sha256: str
    ok: bool
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OllamaBlobIdentity:
    model: str
    blob_sha256: str
    effective_config_sha256: str
    ok: bool
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        mapped = dump()
        if isinstance(mapped, Mapping):
            return mapped
    return {}


def _field(value: object, name: str, default: object = "") -> object:
    mapped = _mapping(value)
    if name in mapped:
        return mapped[name]
    return getattr(value, name, default)


def _blob_ref(show_result: object) -> str:
    modelfile = str(_field(show_result, "modelfile", "") or "")
    from_match = _FROM_RE.search(modelfile)
    source = from_match.group(1) if from_match else modelfile
    digest = _DIGEST_RE.search(source)
    return digest.group(1).lower() if digest else ""


def _modelfile_template(modelfile: str) -> str:
    match = _MODELFILE_TEMPLATE_RE.search(modelfile.replace("\r\n", "\n"))
    if match is None:
        return ""
    template = match.group(1).strip()
    if template.startswith('"""') and template.endswith('"""'):
        template = template[3:-3]
    elif template.startswith('"') and template.endswith('"'):
        template = template[1:-1]
    return template.strip()


def _template(show_result: object) -> str:
    # Ollama may expose the imported base model's large upstream chat template
    # in ``show.template`` even when the local alias's generated Modelfile has
    # an explicit override.  The Modelfile is the effective alias contract and
    # therefore wins when present; the field remains a compatibility fallback.
    modelfile = str(_field(show_result, "modelfile", "") or "")
    template = _modelfile_template(modelfile)
    if not template:
        template = str(_field(show_result, "template", "") or "")
    return template.replace("\r\n", "\n").strip()


def _parameter_values(show_result: object) -> dict[str, list[str]]:
    raw = _field(show_result, "parameters", "")
    values: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            entries = value if isinstance(value, (list, tuple)) else (value,)
            values[str(key).lower()] = [str(item).strip(' "') for item in entries]
        return values
    for line in str(raw or "").splitlines():
        key, separator, value = line.strip().partition(" ")
        if not separator:
            continue
        values.setdefault(key.lower(), []).append(value.strip().strip('"'))
    return values


def _parameters_match(show_result: object) -> bool:
    values = _parameter_values(show_result)
    stops = set(values.get("stop", ()))
    try:
        temperature = float(values.get("temperature", [""])[0])
        top_p = float(values.get("top_p", [""])[0])
        num_ctx = int(float(values.get("num_ctx", [""])[0]))
    except (TypeError, ValueError, IndexError):
        return False
    return bool(
        {"<|im_end|>", "</s>"}.issubset(stops)
        and temperature == 0.7
        and top_p == 0.95
        and num_ctx == 8192
    )


def _effective_config_digest(show_result: object) -> str:
    modelfile = str(_field(show_result, "modelfile", "") or "")
    payload = {
        "modelfile": modelfile.replace("\r\n", "\n").strip(),
        "template": _template(show_result),
        "parameters": _parameter_values(show_result),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _expected_template(modelfile: Path = DEFAULT_MODELFILE) -> str:
    text = modelfile.read_text(encoding="utf-8").replace("\r\n", "\n")
    template = _modelfile_template(text)
    if not template:
        raise ValueError(f"MiniCPM Modelfile has no TEMPLATE block: {modelfile}")
    return template


def verify_minicpm_identity(
    *,
    show: Callable[[str], object] | None = None,
    host: str | None = None,
    alias: str = LOCAL_MODEL,
    source: str = SOURCE_MODEL,
    modelfile: Path = DEFAULT_MODELFILE,
    client_headers: Mapping[str, str] | None = None,
    timeout_sec: float = 5.0,
) -> ModelIdentity:
    """Verify that the local alias is the official Q8 blob plus pinned template."""

    try:
        if show is None:
            import ollama

            client = ollama.Client(
                host=host,
                headers=client_headers or _NO_AMBIENT_AUTH,
                timeout=max(0.1, float(timeout_sec)),
            )
            show = lambda model: client.show(model)  # noqa: E731 - tiny injected adapter
        alias_show = show(alias)
        source_show = show(source)
        alias_blob = _blob_ref(alias_show)
        source_blob = _blob_ref(source_show)
        quantization = str(
            _field(_field(alias_show, "details", {}), "quantization_level", "")
            or ""
        )
        blob_match = bool(alias_blob and source_blob and alias_blob == source_blob)
        pinned_blob_match = bool(
            alias_blob == source_blob == SOURCE_MODEL_BLOB_SHA256
        )
        q8 = quantization.upper() == "Q8_0"
        template_match = _template(alias_show) == _expected_template(modelfile)
        parameters_match = _parameters_match(alias_show)
        effective_config_sha256 = _effective_config_digest(alias_show)
        ok = bool(
            blob_match
            and pinned_blob_match
            and q8
            and template_match
            and parameters_match
            and re.fullmatch(
                r"[0-9a-f]{64}", effective_config_sha256, re.IGNORECASE
            )
            is not None
        )
        return ModelIdentity(
            alias=alias,
            source=source,
            alias_blob_sha256=alias_blob,
            source_blob_sha256=source_blob,
            alias_blob=alias_blob[:12],
            source_blob=source_blob[:12],
            blob_match=blob_match,
            pinned_blob_match=pinned_blob_match,
            quantization=quantization,
            q8=q8,
            template_match=template_match,
            parameters_match=parameters_match,
            effective_config_sha256=effective_config_sha256,
            ok=ok,
        )
    except Exception as exc:
        return ModelIdentity(
            alias=alias,
            source=source,
            alias_blob_sha256="",
            source_blob_sha256="",
            alias_blob="",
            source_blob="",
            blob_match=False,
            pinned_blob_match=False,
            quantization="",
            q8=False,
            template_match=False,
            parameters_match=False,
            effective_config_sha256="",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def verify_ollama_blob_identity(
    model: str,
    *,
    show: Callable[[str], object] | None = None,
    host: str | None = None,
    client_headers: Mapping[str, str] | None = None,
    timeout_sec: float = 5.0,
) -> OllamaBlobIdentity:
    """Pin an Ollama alias's immutable blob and effective prompt parameters."""

    try:
        if show is None:
            import ollama

            client = ollama.Client(
                host=host,
                headers=client_headers or _NO_AMBIENT_AUTH,
                timeout=max(0.1, float(timeout_sec)),
            )
            show = lambda name: client.show(name)  # noqa: E731
        shown = show(model)
        blob = _blob_ref(shown)
        config_digest = _effective_config_digest(shown)
        ok = bool(
            re.fullmatch(r"[0-9a-f]{64}", blob, re.IGNORECASE) is not None
            and re.fullmatch(
                r"[0-9a-f]{64}", config_digest, re.IGNORECASE
            ) is not None
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
