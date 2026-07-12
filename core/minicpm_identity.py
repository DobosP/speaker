"""Canonical MiniCPM5-1B desktop Ollama identity.

The verifier in this module is deliberately transport-free: callers provide an
``ollama show``-compatible function.  Provisioning, native readiness, doctor,
Docker diagnostics, and conversation evaluation therefore share one identity
decision without sharing network or process construction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping


_MODELFILE_TEMPLATE_RE = re.compile(
    r"^TEMPLATE[ \t]+(.*?)"
    r"(?=^(?:FROM|TEMPLATE|PARAMETER|SYSTEM|MESSAGE|ADAPTER|LICENSE)\b|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_DIGEST_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)
_ALLOWED_ALIAS_DIRECTIVES = frozenset({"FROM", "TEMPLATE", "PARAMETER", "LICENSE"})

_CHATML_TEMPLATE = """{{- if .Messages -}}
{{- range .Messages -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end -}}
<|im_start|>assistant
{{ end -}}"""


@dataclass(frozen=True)
class MiniCPMQ8Contract:
    alias: str
    source: str
    blob_sha256: str
    quantization: str
    template: str
    parameters: tuple[tuple[str, tuple[str, ...]], ...]
    modelfile: Path


MINICPM_Q8_CONTRACT = MiniCPMQ8Contract(
    alias="minicpm5-1b:q8",
    source="hf.co/openbmb/MiniCPM5-1B-GGUF:Q8_0",
    blob_sha256=(
        "0dc7638539067268774c275a14a6ec9c7e01f7eeb2cff606c8590361fa527e4c"
    ),
    quantization="Q8_0",
    template=_CHATML_TEMPLATE,
    parameters=(
        ("stop", ("<|im_end|>", "</s>")),
        ("temperature", ("0.7",)),
        ("top_p", ("0.95",)),
        ("num_ctx", ("8192",)),
    ),
    modelfile=(
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "ollama"
        / "Modelfile.minicpm5-1b-q8"
    ),
)


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


def _top_level_directives(modelfile: str) -> tuple[tuple[str, str], ...]:
    """Return directives while ignoring multiline TEMPLATE/LICENSE bodies."""
    rows: list[tuple[str, str]] = []
    block_delimiter = ""
    for line in modelfile.replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        if block_delimiter:
            if block_delimiter in stripped:
                block_delimiter = ""
            continue
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        directive = parts[0]
        if len(parts) == 1:
            rows.append((directive.upper(), ""))
            continue
        value = parts[1]
        rows.append((directive.upper(), value.strip()))
        if value.count('"""') % 2:
            block_delimiter = '"""'
        elif value.startswith('"') and not value.endswith('"'):
            block_delimiter = '"'
    return tuple(rows)


def _from_values(modelfile: str) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for directive, value in _top_level_directives(modelfile)
        if directive == "FROM"
    )


def _blob_ref(show_result: object) -> str:
    modelfile = str(_field(show_result, "modelfile", "") or "")
    sources = _from_values(modelfile)
    if len(sources) != 1:
        return ""
    digest = _DIGEST_RE.search(sources[0])
    return digest.group(1).lower() if digest else ""


def _modelfile_template(modelfile: str) -> str:
    match = _MODELFILE_TEMPLATE_RE.search(modelfile.replace("\r\n", "\n"))
    if match is None:
        return ""
    # Strip syntax whitespace outside the quotes, then preserve every byte
    # inside them (apart from normalized line endings). Boundary whitespace is
    # prompt behavior and therefore part of the exact identity.
    template = match.group(1).strip(" \t\r\n")
    if template.startswith('"""') and template.endswith('"""'):
        template = template[3:-3]
    elif template.startswith('"') and template.endswith('"'):
        template = template[1:-1]
    return template


def _template(show_result: object) -> str:
    # Ollama can expose the imported base template in ``show.template`` while
    # the alias Modelfile carries an explicit override.  The alias Modelfile is
    # the effective contract and wins when present.
    modelfile = str(_field(show_result, "modelfile", "") or "")
    has_explicit_template = any(
        directive == "TEMPLATE"
        for directive, _value in _top_level_directives(modelfile)
    )
    template = (
        _modelfile_template(modelfile)
        if has_explicit_template
        else str(_field(show_result, "template", "") or "")
    )
    return template.replace("\r\n", "\n")


def _clean_parameter(value: object) -> str:
    return str(value).strip().strip('"')


def _parameter_values(show_result: object) -> dict[str, list[str]]:
    raw = _field(show_result, "parameters", "")
    values: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            entries = value if isinstance(value, (list, tuple)) else (value,)
            values[str(key).lower()] = [_clean_parameter(item) for item in entries]
        return values
    for line in str(raw or "").splitlines():
        key, separator, value = line.strip().partition(" ")
        if separator:
            values.setdefault(key.lower(), []).append(_clean_parameter(value))
    return values


def _modelfile_parameter_values(modelfile: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for directive, raw in _top_level_directives(modelfile):
        if directive != "PARAMETER":
            continue
        parts = raw.split(None, 1)
        if len(parts) == 2:
            values.setdefault(parts[0].lower(), []).append(
                _clean_parameter(parts[1])
            )
    return values


def _expected_parameters(
    contract: MiniCPMQ8Contract = MINICPM_Q8_CONTRACT,
) -> dict[str, list[str]]:
    return {key: list(values) for key, values in contract.parameters}


def _one_float(values: dict[str, list[str]], key: str) -> float | None:
    entries = values.get(key, ())
    if len(entries) != 1:
        return None
    try:
        value = float(entries[0])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parameters_match_values(
    values: dict[str, list[str]],
    contract: MiniCPMQ8Contract = MINICPM_Q8_CONTRACT,
) -> bool:
    expected = _expected_parameters(contract)
    if set(values) != set(expected):
        return False
    if Counter(values["stop"]) != Counter(expected["stop"]):
        return False
    temperature = _one_float(values, "temperature")
    top_p = _one_float(values, "top_p")
    num_ctx = _one_float(values, "num_ctx")
    return bool(
        temperature == float(expected["temperature"][0])
        and top_p == float(expected["top_p"][0])
        and num_ctx == float(expected["num_ctx"][0])
        and num_ctx is not None
        and num_ctx.is_integer()
    )


def _parameters_match(
    show_result: object,
    contract: MiniCPMQ8Contract = MINICPM_Q8_CONTRACT,
) -> bool:
    return _parameters_match_values(_parameter_values(show_result), contract)


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


def _unexpected_modelfile_directives(modelfile: str) -> tuple[str, ...]:
    return tuple(
        directive
        for directive, _value in _top_level_directives(modelfile)
        if directive not in _ALLOWED_ALIAS_DIRECTIVES
    )


def _expected_template() -> str:
    return MINICPM_Q8_CONTRACT.template


def validate_minicpm_modelfile(
    path: Path = MINICPM_Q8_CONTRACT.modelfile,
) -> None:
    """Prove the provisioned file renders the canonical identity contract."""
    if not path.is_file():
        raise FileNotFoundError(f"MiniCPM Modelfile not found: {path}")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    sources = _from_values(text)
    if len(sources) != 1:
        raise ValueError(
            f"MiniCPM Modelfile must have exactly one FROM; found {len(sources)}"
        )
    if sources[0] != MINICPM_Q8_CONTRACT.source:
        raise ValueError(
            "MiniCPM Modelfile FROM does not match the pinned source "
            f"{MINICPM_Q8_CONTRACT.source}"
        )
    template_count = sum(
        directive == "TEMPLATE" for directive, _value in _top_level_directives(text)
    )
    if template_count != 1:
        raise ValueError(
            "MiniCPM Modelfile must have exactly one TEMPLATE; "
            f"found {template_count}"
        )
    if _modelfile_template(text) != MINICPM_Q8_CONTRACT.template:
        raise ValueError("MiniCPM Modelfile TEMPLATE does not match the canonical template")
    if not _parameters_match_values(_modelfile_parameter_values(text)):
        raise ValueError("MiniCPM Modelfile PARAMETER set is not canonical")
    unexpected = _unexpected_modelfile_directives(text)
    if unexpected:
        raise ValueError(
            "MiniCPM Modelfile has unsupported directives: "
            + ", ".join(unexpected)
        )


def is_minicpm_model_name(model: str) -> bool:
    """Whether an Ollama model name claims to be a MiniCPM identity."""
    return "minicpm" in str(model).casefold()


def verify_minicpm_q8_identity(
    *,
    show: Callable[[str], object],
) -> ModelIdentity:
    """Verify the one supported desktop Q8 alias; never performs I/O itself."""
    contract = MINICPM_Q8_CONTRACT
    try:
        alias_show = show(contract.alias)
        source_show = show(contract.source)
        alias_modelfile = str(_field(alias_show, "modelfile", "") or "")
        source_modelfile = str(_field(source_show, "modelfile", "") or "")
        alias_sources = _from_values(alias_modelfile)
        source_sources = _from_values(source_modelfile)
        alias_blob = _blob_ref(alias_show)
        source_blob = _blob_ref(source_show)
        quantization = str(
            _field(_field(alias_show, "details", {}), "quantization_level", "")
            or ""
        )
        blob_match = bool(alias_blob and source_blob and alias_blob == source_blob)
        pinned_blob_match = bool(
            alias_blob == source_blob == contract.blob_sha256
        )
        q8 = quantization.upper() == contract.quantization
        template_match = _template(alias_show) == contract.template
        parameters_match = _parameters_match(alias_show, contract)
        effective_config_sha256 = _effective_config_digest(alias_show)
        alias_template_count = sum(
            directive == "TEMPLATE"
            for directive, _value in _top_level_directives(alias_modelfile)
        )
        unexpected = _unexpected_modelfile_directives(alias_modelfile)

        failures: list[str] = []
        if len(alias_sources) != 1:
            failures.append(
                f"alias Modelfile has {len(alias_sources)} FROM directives, expected 1"
            )
        if len(source_sources) != 1:
            failures.append(
                f"source Modelfile has {len(source_sources)} FROM directives, expected 1"
            )
        if alias_template_count > 1:
            failures.append(
                "alias Modelfile has multiple TEMPLATE directives"
            )
        if not alias_blob:
            failures.append("alias blob digest missing")
        if not source_blob:
            failures.append("source blob digest missing")
        if not blob_match:
            failures.append("alias/source blobs differ")
        if not pinned_blob_match:
            failures.append("blob is not the pinned official Q8 artifact")
        if not q8:
            failures.append(
                f"quantization is {quantization or 'missing'}, expected {contract.quantization}"
            )
        if not template_match:
            failures.append("alias template differs from the canonical ChatML template")
        if not parameters_match:
            failures.append("alias parameters differ from the canonical exact set")
        if unexpected:
            failures.append(
                "alias has unsupported Modelfile directives: "
                + ", ".join(unexpected)
            )

        return ModelIdentity(
            alias=contract.alias,
            source=contract.source,
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
            ok=not failures,
            error="; ".join(failures),
        )
    except Exception as exc:
        return ModelIdentity(
            alias=contract.alias,
            source=contract.source,
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
