"""Shared, privacy-safe release provenance for real Ollama evidence.

The conversation and memory gates both need to bind a result to one clean Git
revision and stable model identities.  Keep that contract here rather than in a
CLI module so independent evaluators cannot silently drift.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping
from urllib.parse import urlparse

from tools.setup_minicpm import (
    LOCAL_MODEL,
    SOURCE_MODEL,
    SOURCE_MODEL_BLOB_SHA256,
)

from .identity import verify_minicpm_identity, verify_ollama_blob_identity


_REPO = Path(__file__).resolve().parents[2]
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"
LOCAL_OLLAMA_HEADERS = {"authorization": "Bearer speaker-local-evaluation"}
IdentityRecord = Callable[[str, dict], dict[str, object]]


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def repository_metadata(root: Path = _REPO) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "", "dirty": None}


def config_metadata(config: dict, *, include_local_config: bool) -> dict[str, object]:
    llm = config.get("llm", {}) or {}
    raw_host = str(llm.get("host", "") or "")
    parsed_host = urlparse(raw_host if "://" in raw_host else f"http://{raw_host}")
    hostname = parsed_host.hostname or ""
    try:
        port = parsed_host.port
    except ValueError:
        port = None
    display_hostname = f"[{hostname}]" if ":" in hostname else hostname
    host_label = f"{parsed_host.scheme or 'http'}://{display_hostname}"
    if port is not None:
        host_label += f":{port}"
    return {
        "include_local_config": include_local_config,
        "contract_sha256": json_sha256(config),
        "llm_options_sha256": json_sha256(llm.get("options")),
        "backend": str(llm.get("backend", "ollama")),
        "host": host_label,
        "keep_alive": llm.get("keep_alive"),
        "configured_role_models": {
            "main": str(llm.get("main_model", "")),
            "fast": str(llm.get("fast_model", "")),
        },
    }


def model_identity_record(
    model: str,
    config: dict,
    *,
    minicpm_verifier=verify_minicpm_identity,
    generic_verifier=verify_ollama_blob_identity,
) -> dict[str, object]:
    host = str((config.get("llm", {}) or {}).get("host", "") or "") or None
    if model == LOCAL_MODEL:
        identity = minicpm_verifier(host=host)
        return {
            "model": model,
            "verification": "minicpm_q8_blob_template_parameters",
            "required": True,
            **identity.as_dict(),
        }
    identity = generic_verifier(model, host=host)
    return {
        "model": model,
        "verification": "ollama_blob_effective_config",
        "required": True,
        **identity.as_dict(),
    }


def identity_contract(record: object) -> tuple[str, str] | None:
    if not isinstance(record, dict) or record.get("ok") is not True:
        return None
    blob = str(record.get("alias_blob_sha256") or record.get("blob_sha256") or "").lower()
    effective = str(record.get("effective_config_sha256", "") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", blob) is None:
        return None
    if re.fullmatch(r"[0-9a-f]{64}", effective) is None:
        return None
    return blob, effective


def full_identity_contract(
    record: object,
    model: str,
) -> tuple[str, str] | None:
    """Validate one complete conversation/memory release identity record."""
    if not isinstance(record, Mapping):
        return None
    if (
        str(record.get("model", "")) != model
        or record.get("required") is not True
        or record.get("ok") is not True
    ):
        return None
    verification = str(record.get("verification", ""))
    effective = str(record.get("effective_config_sha256", "") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", effective) is None:
        return None
    if model == LOCAL_MODEL:
        alias_blob = str(record.get("alias_blob_sha256", "") or "").lower()
        source_blob = str(record.get("source_blob_sha256", "") or "").lower()
        if not bool(
            verification == "minicpm_q8_blob_template_parameters"
            and record.get("alias") == LOCAL_MODEL
            and record.get("source") == SOURCE_MODEL
            and alias_blob == source_blob == SOURCE_MODEL_BLOB_SHA256
            and record.get("blob_match") is True
            and record.get("pinned_blob_match") is True
            and record.get("q8") is True
            and record.get("template_match") is True
            and record.get("parameters_match") is True
        ):
            return None
        return alias_blob, effective
    blob = str(record.get("blob_sha256", "") or "").lower()
    if not bool(
        verification == "ollama_blob_effective_config"
        and re.fullmatch(r"[0-9a-f]{64}", blob)
    ):
        return None
    return blob, effective


def identity_snapshot(
    role_models: dict[str, str],
    config: dict,
    *,
    record_fn: IdentityRecord | None = None,
) -> dict[str, object]:
    recorder = record_fn or model_identity_record
    models = {
        model: recorder(model, config)
        for model in sorted(set(role_models.values()))
        if model
    }
    expected = {model for model in role_models.values() if model}
    return {
        "role_models": dict(role_models),
        "models": models,
        "ok": bool(
            models
            and set(models) == expected
            and all(
                full_identity_contract(record, model) is not None
                for model, record in models.items()
            )
        ),
    }


def identity_bundle(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    before_models = before.get("models", {})
    after_models = after.get("models", {})
    stable = bool(
        isinstance(before_models, dict)
        and isinstance(after_models, dict)
        and set(before_models) == set(after_models)
        and all(
            full_identity_contract(before_models[model], model)
            == full_identity_contract(after_models[model], model)
            is not None
            for model in before_models
        )
    )
    return {
        "before": before,
        "after": after,
        "stable": stable,
        "ok": bool(before.get("ok") is True and after.get("ok") is True and stable),
    }


def validate_identity_bundle(
    role_models: object,
    bundle: object,
) -> tuple[bool, dict[str, tuple[str, str]]]:
    """Recompute full role/model identity validity without trusting producer flags."""
    if not isinstance(bundle, Mapping) or not isinstance(role_models, Mapping):
        return False, {}
    before = bundle.get("before")
    after = bundle.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False, {}
    expected_roles = {
        "main": str(role_models.get("main", "")),
        "fast": str(role_models.get("fast", "")),
    }
    if not all(expected_roles.values()):
        return False, {}
    if (
        before.get("role_models") != expected_roles
        or after.get("role_models") != expected_roles
    ):
        return False, {}
    before_models = before.get("models")
    after_models = after.get("models")
    expected_models = set(expected_roles.values())
    if not isinstance(before_models, Mapping) or not isinstance(after_models, Mapping):
        return False, {}
    if set(before_models) != expected_models or set(after_models) != expected_models:
        return False, {}
    contracts: dict[str, tuple[str, str]] = {}
    for model in expected_models:
        before_contract = full_identity_contract(before_models.get(model), model)
        after_contract = full_identity_contract(after_models.get(model), model)
        if before_contract is None or before_contract != after_contract:
            return False, {}
        contracts[model] = before_contract
    valid = bool(
        before.get("ok") is True
        and after.get("ok") is True
        and bundle.get("ok") is True
        and bundle.get("stable") is True
    )
    return valid, contracts


def failed_identity_records(snapshot: dict[str, object]) -> dict[str, object]:
    records = snapshot.get("models", {})
    if not isinstance(records, dict):
        return {"snapshot": snapshot}
    return {
        model: record
        for model, record in records.items()
        if full_identity_contract(record, model) is None
    }
