from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import tools.conversation_eval.__main__ as conversation_cli
from tools.conversation_eval.identity import _expected_template, verify_minicpm_identity
from tools.conversation_eval.report import build_report
from tools.conversation_eval.schema import ScenarioResult
from tools.conversation_eval.scenarios import SCENARIOS
from tools.setup_minicpm import (
    LOCAL_MODEL,
    SOURCE_MODEL,
    SOURCE_MODEL_BLOB_SHA256,
)


def _suite(model: str) -> tuple[ScenarioResult, ...]:
    return tuple(
        ScenarioResult(
            scenario_id=scenario.scenario_id,
            description=scenario.description,
            model=model,
            run_index=run_index,
            passed=True,
            duration_ms=1.0,
            turns=(),
            checks=(),
            trace=(),
            memory_before=(),
            memory_after=(),
            metrics=(),
            model_calls=(),
        )
        for scenario in SCENARIOS
        for run_index in (1, 2, 3)
    )


def _generic_record(model: str, blob: str, effective: str) -> dict[str, object]:
    return {
        "model": model,
        "verification": "ollama_blob_effective_config",
        "required": True,
        "blob_sha256": blob,
        "effective_config_sha256": effective,
        "ok": True,
        "error": "",
    }


def _snapshot(
    role_models: dict[str, str],
    contracts: dict[str, tuple[str, str]],
) -> dict[str, object]:
    return {
        "role_models": role_models,
        "models": {
            model: _generic_record(model, *contracts[model])
            for model in set(role_models.values())
        },
        "ok": True,
    }


def _bundle(
    role_models: dict[str, str],
    contracts: dict[str, tuple[str, str]],
) -> dict[str, object]:
    before = _snapshot(role_models, contracts)
    return {
        "before": before,
        "after": deepcopy(before),
        "stable": True,
        "ok": True,
    }


def _evaluation(role_models: dict[str, str], *, assignment: str = "production_hybrid_fast_override") -> dict[str, object]:
    roles_by_model: dict[str, list[str]] = {}
    for role, model in role_models.items():
        roles_by_model.setdefault(model, []).append(role)
    return {
        "model_assignment": assignment,
        "role_models": role_models,
        "warmup": {
            "policy": "production_system_prompt",
            "performed": True,
            "gate_eligible": True,
            "ok": True,
            "system_prompt_sha256": "9" * 64,
            "calls": [
                {"model": model, "roles": roles, "ok": True}
                for model, roles in roles_by_model.items()
            ],
        },
    }


def _metadata(
    *,
    configured_main: str = "main:test",
    topology: str = "production-hybrid",
    include_local_config: bool = False,
) -> dict[str, object]:
    repository = {"revision": "1" * 40, "dirty": False}
    config = {
        "contract_sha256": "2" * 64,
        "llm_options_sha256": "3" * 64,
        "include_local_config": include_local_config,
        "backend": "ollama",
        "host": "http://127.0.0.1:11434",
        "keep_alive": -1,
        "configured_role_models": {
            "main": configured_main,
            "fast": "candidate:test",
        },
    }
    return {
        "repository": repository,
        "config": config,
        "provenance_snapshots": {
            "before": {
                "repository": deepcopy(repository),
                "config": deepcopy(config),
            },
            "after": {
                "repository": deepcopy(repository),
                "config": deepcopy(config),
            },
        },
        "topology": topology,
        "warm_policy": "production",
        "execution_order": ["baseline", "candidate"],
        "ollama_python_version": "0.test",
    }


def _valid_report_inputs() -> dict[str, object]:
    main = "main:test"
    candidate = "candidate:test"
    baseline = "baseline:test"
    candidate_roles = {"main": main, "fast": candidate}
    baseline_roles = {"main": main, "fast": baseline}
    common = ("a" * 64, "b" * 64)
    candidate_contracts = {main: common, candidate: ("c" * 64, "d" * 64)}
    baseline_contracts = {main: common, baseline: ("e" * 64, "f" * 64)}
    return {
        "mode": "ollama",
        "device": "desktop_gpu_4090",
        "candidate_results": _suite(candidate),
        "baseline_results": _suite(baseline),
        "identities": {
            "candidate": _bundle(candidate_roles, candidate_contracts),
            "baseline": _bundle(baseline_roles, baseline_contracts),
        },
        "metadata": _metadata(),
        "candidate_metadata": _evaluation(candidate_roles),
        "baseline_metadata": _evaluation(baseline_roles),
        "provenance_ok": True,
    }


def test_real_report_accepts_only_stable_clean_full_role_evidence():
    report = build_report(**_valid_report_inputs())

    assert report["gate"]["repository_stable"] is True
    assert report["gate"]["config_stable"] is True
    assert report["gate"]["identity_evidence_ok"] is True
    assert report["gate"]["shared_main_contract_equal"] is True
    assert report["gate"]["ab_distinct"] is True
    assert report["gate"]["passed"] is True


@pytest.mark.parametrize("mutation", ("dirty", "revision", "config", "local"))
def test_real_report_rejects_repository_or_effective_config_drift(mutation: str):
    inputs = _valid_report_inputs()
    metadata = deepcopy(inputs["metadata"])
    inputs["metadata"] = metadata
    if mutation == "dirty":
        metadata["provenance_snapshots"]["before"]["repository"]["dirty"] = True
    elif mutation == "revision":
        metadata["provenance_snapshots"]["before"]["repository"]["revision"] = "4" * 40
    elif mutation == "config":
        metadata["provenance_snapshots"]["before"]["config"]["contract_sha256"] = "5" * 64
    else:
        for snapshot in ("before", "after"):
            metadata["provenance_snapshots"][snapshot]["config"]["include_local_config"] = True
        metadata["config"]["include_local_config"] = True

    gate = build_report(**inputs)["gate"]

    assert gate["provenance_ok"] is False
    assert gate["passed"] is False
    if mutation == "local":
        assert gate["local_config_excluded"] is False


def test_real_report_rejects_identity_change_or_missing_effective_role():
    changed_inputs = _valid_report_inputs()
    changed = changed_inputs["identities"]["candidate"]
    changed["after"]["models"]["candidate:test"]["blob_sha256"] = "6" * 64
    changed_gate = build_report(**changed_inputs)["gate"]

    missing_inputs = _valid_report_inputs()
    missing = missing_inputs["identities"]["candidate"]
    del missing["before"]["models"]["main:test"]
    del missing["after"]["models"]["main:test"]
    missing_gate = build_report(**missing_inputs)["gate"]

    assert changed_gate["identity_evidence_ok"] is False
    assert changed_gate["passed"] is False
    assert missing_gate["identity_evidence_ok"] is False
    assert missing_gate["passed"] is False


def test_real_report_requires_distinct_fast_blob_not_alias_or_config():
    inputs = _valid_report_inputs()
    candidate_fast = inputs["identities"]["candidate"]
    baseline_fast = inputs["identities"]["baseline"]
    same_contract = deepcopy(
        candidate_fast["before"]["models"]["candidate:test"]
    )
    same_contract["model"] = "baseline:test"
    same_contract["effective_config_sha256"] = "7" * 64
    for phase in ("before", "after"):
        baseline_fast[phase]["models"]["baseline:test"] = deepcopy(same_contract)

    gate = build_report(**inputs)["gate"]

    assert gate["identity_evidence_ok"] is True
    assert gate["ab_distinct"] is False
    assert gate["ab_valid"] is False
    assert gate["passed"] is False


def test_real_report_binds_main_to_config_and_rejects_all_role_adoption():
    wrong_main_inputs = _valid_report_inputs()
    wrong_main_inputs["metadata"] = _metadata(configured_main="other-main:test")
    wrong_main_gate = build_report(**wrong_main_inputs)["gate"]

    all_role_inputs = _valid_report_inputs()
    all_role_inputs["metadata"] = _metadata(topology="all-roles")
    all_role_inputs["candidate_metadata"] = _evaluation(
        {"main": "candidate:test", "fast": "candidate:test"},
        assignment="all_roles_override",
    )
    all_role_inputs["baseline_metadata"] = _evaluation(
        {"main": "baseline:test", "fast": "baseline:test"},
        assignment="all_roles_override",
    )
    all_role_inputs["identities"] = {
        "candidate": _bundle(
            {"main": "candidate:test", "fast": "candidate:test"},
            {"candidate:test": ("c" * 64, "d" * 64)},
        ),
        "baseline": _bundle(
            {"main": "baseline:test", "fast": "baseline:test"},
            {"baseline:test": ("e" * 64, "f" * 64)},
        ),
    }
    all_role_gate = build_report(**all_role_inputs)["gate"]

    assert wrong_main_gate["topology_ok"] is False
    assert wrong_main_gate["passed"] is False
    assert all_role_gate["adoption_topology_eligible"] is False
    assert all_role_gate["topology_ok"] is False
    assert all_role_gate["passed"] is False


def test_minicpm_identity_retains_full_pinned_blob_and_effective_contract():
    digest = SOURCE_MODEL_BLOB_SHA256
    alias = {
        "modelfile": f"FROM /models/sha256-{digest}",
        "template": _expected_template(),
        "parameters": (
            'stop "<|im_end|>"\n'
            'stop "</s>"\n'
            "temperature 0.7\n"
            "top_p 0.95\n"
            "num_ctx 8192"
        ),
        "details": {"quantization_level": "Q8_0"},
    }
    source = {"modelfile": f"FROM /models/sha256-{digest}"}

    identity = verify_minicpm_identity(
        show=lambda model: alias if model == LOCAL_MODEL else source
    )

    assert identity.ok is True
    assert identity.alias_blob_sha256 == digest
    assert identity.source_blob_sha256 == digest
    assert len(identity.effective_config_sha256) == 64


def test_identity_snapshot_deduplicates_shared_roles(monkeypatch):
    calls: list[str] = []

    def record(model: str, _config: dict) -> dict[str, object]:
        calls.append(model)
        return _generic_record(model, "a" * 64, "b" * 64)

    monkeypatch.setattr(conversation_cli, "_model_identity_record", record)

    snapshot = conversation_cli._identity_snapshot(
        {"main": "shared:test", "fast": "shared:test"},
        {},
    )

    assert calls == ["shared:test"]
    assert snapshot["ok"] is True
