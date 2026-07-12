from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Mapping

from tools.setup_minicpm import (
    DEFAULT_MODELFILE,
    LOCAL_MODEL,
    SOURCE_MODEL,
    SOURCE_MODEL_BLOB_SHA256,
)


_TEMPLATE_RE = re.compile(r'TEMPLATE\s+"""(.*?)"""', re.DOTALL)
_FROM_RE = re.compile(r"^FROM\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_DIGEST_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)
_NO_AMBIENT_AUTH = {"authorization": "Bearer speaker-local-evaluation"}


@dataclass(frozen=True)
class ModelIdentity:
    alias: str
    source: str
    alias_blob: str
    source_blob: str
    blob_match: bool
    pinned_blob_match: bool
    quantization: str
    q8: bool
    template_match: bool
    parameters_match: bool
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


def _template(show_result: object) -> str:
    template = str(_field(show_result, "template", "") or "")
    if not template:
        modelfile = str(_field(show_result, "modelfile", "") or "")
        match = _TEMPLATE_RE.search(modelfile)
        template = match.group(1) if match else ""
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


def _expected_template(modelfile: Path = DEFAULT_MODELFILE) -> str:
    text = modelfile.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = _TEMPLATE_RE.search(text)
    if match is None:
        raise ValueError(f"MiniCPM Modelfile has no TEMPLATE block: {modelfile}")
    return match.group(1).strip()


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
        ok = bool(
            blob_match
            and pinned_blob_match
            and q8
            and template_match
            and parameters_match
        )
        return ModelIdentity(
            alias=alias,
            source=source,
            alias_blob=alias_blob[:12],
            source_blob=source_blob[:12],
            blob_match=blob_match,
            pinned_blob_match=pinned_blob_match,
            quantization=quantization,
            q8=q8,
            template_match=template_match,
            parameters_match=parameters_match,
            ok=ok,
        )
    except Exception as exc:
        return ModelIdentity(
            alias=alias,
            source=source,
            alias_blob="",
            source_blob="",
            blob_match=False,
            pinned_blob_match=False,
            quantization="",
            q8=False,
            template_match=False,
            parameters_match=False,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
