from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlparse

from core.minicpm_identity import MINICPM_Q8_CONTRACT

from .scenarios import SCENARIOS, SCENARIO_SET_VERSION
from .schema import ModelSummary, SCHEMA_VERSION, ScenarioResult


LOCAL_MODEL = MINICPM_Q8_CONTRACT.alias
SOURCE_MODEL = MINICPM_Q8_CONTRACT.source
SOURCE_MODEL_BLOB_SHA256 = MINICPM_Q8_CONTRACT.blob_sha256


_STRUCTURAL_KEYS: dict[str, tuple[str, ...]] = {
    "intent.decision": ("kind", "reason"),
    "task.started": ("mode", "capability"),
    "task.completed": ("mode", "capability"),
    "task.cancelled": ("mode",),
    "task.failed": ("mode",),
    "capability.started": ("invocation_id", "name", "planner_tool"),
    "capability.finished": ("invocation_id", "name", "planner_tool"),
    "playback.started": ("fragment_id",),
    "playback.attributed": (
        "fragment_id",
        "input_generation",
        "auxiliary_tts",
    ),
    "playback.terminal": ("fragment_id", "outcome"),
    "memory.commit": ("source", "is_followup"),
    "control.stop": ("reason", "already_cancelled"),
}


def structural_trace(result: ScenarioResult) -> list[dict[str, object]]:
    """Timing/text-independent partial-order trajectory for deterministic A/B.

    Worker, bus, and playback callbacks can legitimately interleave.  Canonical
    ordering therefore uses typed identities within each family (turn/task,
    invocation, fragment) instead of wall-clock observation order.
    """

    rows: list[dict[str, object]] = []
    ordinals: dict[str, int] = defaultdict(int)
    for event in result.trace:
        keys = _STRUCTURAL_KEYS.get(event.kind)
        if keys is None:
            continue
        payload = {key: event.payload.get(key) for key in keys if key in event.payload}
        if event.kind == "capability.finished":
            nested = event.payload.get("result")
            if isinstance(nested, dict):
                payload["ok"] = nested.get("ok")
                data = nested.get("data")
                if isinstance(data, dict) and "route" in data:
                    payload["route"] = data["route"]
        if event.kind == "task.completed":
            data = event.payload.get("data")
            if isinstance(data, dict) and "route" in data:
                payload["route"] = data["route"]
        ordinals[event.kind] += 1
        family = event.kind.split(".", 1)[0]
        family_order = {
            "intent": 0,
            "task": 1,
            "capability": 2,
            "playback": 3,
            "memory": 4,
            "control": 5,
        }.get(family, 9)
        if family == "task":
            identity = event.task_id
            phase = 0 if event.kind == "task.started" else 1
        elif family == "capability":
            identity = f"I{int(payload.get('invocation_id', 0)):06d}"
            phase = 0 if event.kind == "capability.started" else 1
        elif family == "playback":
            identity = str(payload.get("fragment_id", ""))
            phase = 0 if event.kind == "playback.started" else 1
        else:
            identity = f"{event.turn_id or 0}:{ordinals[event.kind]:06d}"
            phase = 0
        rows.append(
            {
                "_sort": (family_order, identity, phase, event.kind),
                "kind": event.kind,
                "task_id": event.task_id,
                "turn_id": event.turn_id,
                "payload": payload,
            }
        )
    rows.sort(key=lambda row: row["_sort"])
    for row in rows:
        row.pop("_sort", None)
    return rows


def _trace_signature(result: ScenarioResult) -> str:
    encoded = json.dumps(structural_trace(result), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _first_trace_difference(
    baseline: ScenarioResult,
    candidate: ScenarioResult,
) -> dict[str, object] | None:
    left = structural_trace(baseline)
    right = structural_trace(candidate)
    for index in range(max(len(left), len(right))):
        before = left[index] if index < len(left) else None
        after = right[index] if index < len(right) else None
        if before != after:
            return {"index": index, "baseline": before, "candidate": after}
    return None


def summarize_model(model: str, results: tuple[ScenarioResult, ...]) -> ModelSummary:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.scenario_id].append(result)
    for values in grouped.values():
        values.sort(key=lambda result: result.run_index)
    reliability = {
        scenario_id: {
            "runs": len(values),
            "runs_passed": sum(1 for value in values if value.passed),
            "pass_at_1": bool(values and values[0].passed),
            "pass_power_k": bool(values and all(value.passed for value in values)),
        }
        for scenario_id, values in sorted(grouped.items())
    }
    runs = max((len(values) for values in grouped.values()), default=0)
    first_run_passed = bool(
        grouped
        and all(values and values[0].passed for values in grouped.values())
    )
    all_passed = bool(results and all(result.passed for result in results))
    return ModelSummary(
        model=model,
        runs=runs,
        scenarios=len(grouped),
        passed_results=sum(1 for result in results if result.passed),
        total_results=len(results),
        pass_at_1=first_run_passed,
        pass_power_k=all_passed,
        scenario_reliability=reliability,
    )


def compare_results(
    baseline: tuple[ScenarioResult, ...],
    candidate: tuple[ScenarioResult, ...],
) -> dict[str, object]:
    def grouped(results: tuple[ScenarioResult, ...]) -> dict[str, list[ScenarioResult]]:
        output: dict[str, list[ScenarioResult]] = defaultdict(list)
        for result in results:
            output[result.scenario_id].append(result)
        return output

    left = grouped(baseline)
    right = grouped(candidate)
    rows: list[dict[str, object]] = []
    for scenario_id in sorted(set(left) | set(right)):
        baseline_values = sorted(left.get(scenario_id, []), key=lambda item: item.run_index)
        candidate_values = sorted(right.get(scenario_id, []), key=lambda item: item.run_index)
        paired = list(zip(baseline_values, candidate_values))
        trace_differences = [
            {
                "run_index": before.run_index,
                "first_difference": difference,
            }
            for before, after in paired
            if (difference := _first_trace_difference(before, after)) is not None
        ]
        rows.append(
            {
                "scenario_id": scenario_id,
                "baseline_passed": sum(value.passed for value in baseline_values),
                "candidate_passed": sum(value.passed for value in candidate_values),
                "runs": max(len(baseline_values), len(candidate_values)),
                "candidate_regression": bool(
                    baseline_values
                    and all(value.passed for value in baseline_values)
                    and not all(value.passed for value in candidate_values)
                ),
                "baseline_trace_signatures": [
                    _trace_signature(value) for value in baseline_values
                ],
                "candidate_trace_signatures": [
                    _trace_signature(value) for value in candidate_values
                ],
                "structural_trace_equal_runs": len(paired) - len(trace_differences),
                "trace_differences": trace_differences,
            }
        )
    return {
        "rows": rows,
        "regressions": [row["scenario_id"] for row in rows if row["candidate_regression"]],
    }


def _coverage(results: tuple[ScenarioResult, ...]) -> dict[str, object]:
    expected_scenarios = tuple(scenario.scenario_id for scenario in SCENARIOS)
    expected_runs = (1, 2, 3)
    observed: dict[str, list[int]] = defaultdict(list)
    for result in results:
        observed[result.scenario_id].append(int(result.run_index))
    observed_scenarios = tuple(sorted(observed))
    exact = bool(
        observed_scenarios == tuple(sorted(expected_scenarios))
        and all(
            tuple(sorted(observed[scenario_id])) == expected_runs
            for scenario_id in expected_scenarios
        )
    )
    return {
        "required_scenarios": list(expected_scenarios),
        "required_runs": list(expected_runs),
        "observed": {
            scenario_id: sorted(run_indices)
            for scenario_id, run_indices in sorted(observed.items())
        },
        "ok": exact,
    }


def build_report(
    *,
    mode: str,
    device: str,
    candidate_results: tuple[ScenarioResult, ...],
    baseline_results: tuple[ScenarioResult, ...] = (),
    identities: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    candidate_metadata: dict[str, object] | None = None,
    baseline_metadata: dict[str, object] | None = None,
    provenance_ok: bool | None = None,
    diagnostic_override_used: bool = False,
) -> dict[str, object]:
    candidate_model = candidate_results[0].model if candidate_results else ""
    candidate_model_consistent = bool(
        candidate_results
        and all(result.model == candidate_model for result in candidate_results)
    )
    candidate_summary = summarize_model(candidate_model, candidate_results)
    candidate_coverage = _coverage(candidate_results)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "scenario_set_version": SCENARIO_SET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "device": device,
        "scope": (
            "Device-free conversation semantics only; no ASR, echo cancellation, "
            "microphone, speaker, or bare-speaker validation."
        ),
        "metadata": metadata or {},
        "identities": identities or {},
        "candidate": {
            "summary": candidate_summary.as_dict(),
            "coverage": candidate_coverage,
            "evaluation": candidate_metadata or {},
            "results": [result.as_dict() for result in candidate_results],
        },
    }
    baseline_required = mode == "ollama"
    baseline_present = bool(baseline_results)
    baseline_valid = not baseline_required or baseline_present
    baseline_coverage: dict[str, object] | None = None
    baseline_model_consistent = not baseline_required
    no_candidate_regression = True
    if baseline_results:
        baseline_model = baseline_results[0].model
        baseline_model_consistent = all(
            result.model == baseline_model for result in baseline_results
        )
        baseline_summary = summarize_model(baseline_model, baseline_results)
        baseline_coverage = _coverage(baseline_results)
        comparison = compare_results(baseline_results, candidate_results)
        baseline_valid = baseline_summary.pass_power_k
        no_candidate_regression = not bool(comparison["regressions"])
        report["baseline"] = {
            "summary": baseline_summary.as_dict(),
            "coverage": baseline_coverage,
            "evaluation": baseline_metadata or {},
            "results": [result.as_dict() for result in baseline_results],
        }
        report["comparison"] = comparison
    candidate_roles = (
        candidate_metadata.get("role_models", {})
        if isinstance(candidate_metadata, dict)
        else {}
    )
    baseline_roles = (
        baseline_metadata.get("role_models", {})
        if isinstance(baseline_metadata, dict)
        else {}
    )
    identity_records = identities or {}

    def identity_contract(
        record: object,
        model: str,
    ) -> tuple[str, str] | None:
        if not isinstance(record, Mapping):
            return None
        if (
            str(record.get("model", "")) != model
            or record.get("required") is not True
            or record.get("ok") is not True
        ):
            return None
        verification = str(record.get("verification", ""))
        effective = str(record.get("effective_config_sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", effective) is None:
            return None
        if model == LOCAL_MODEL:
            alias_blob = str(record.get("alias_blob_sha256", "")).lower()
            source_blob = str(record.get("source_blob_sha256", "")).lower()
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
        blob = str(record.get("blob_sha256", "")).lower()
        if not bool(
            verification == "ollama_blob_effective_config"
            and re.fullmatch(r"[0-9a-f]{64}", blob)
        ):
            return None
        return blob, effective

    def suite_identity_evidence(
        label: str,
        role_models: object,
    ) -> tuple[bool, dict[str, tuple[str, str]]]:
        bundle = identity_records.get(label)
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
        if before.get("role_models") != expected_roles or after.get("role_models") != expected_roles:
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
            before_contract = identity_contract(before_models.get(model), model)
            after_contract = identity_contract(after_models.get(model), model)
            if before_contract is None or before_contract != after_contract:
                return False, {}
            contracts[model] = before_contract
        return bool(bundle.get("ok") is True and bundle.get("stable") is True), contracts

    candidate_identity_ok, candidate_contracts = suite_identity_evidence(
        "candidate",
        candidate_roles,
    )
    baseline_identity_ok, baseline_contracts = suite_identity_evidence(
        "baseline",
        baseline_roles,
    )
    identity_evidence_ok = bool(
        mode != "ollama"
        or (
            baseline_results
            and candidate_identity_ok
            and baseline_identity_ok
        )
    )
    provenance_asserted = provenance_ok is not None
    effective_provenance_ok = bool(
        (mode != "ollama" and provenance_ok is None)
        or (
            provenance_asserted
            and provenance_ok
            and identity_evidence_ok
        )
    )

    def evaluation_evidence(details: dict[str, object] | None) -> bool:
        if not isinstance(details, dict):
            return False
        role_models = details.get("role_models")
        warmup = details.get("warmup")
        if not isinstance(role_models, Mapping) or not isinstance(warmup, Mapping):
            return False
        calls = warmup.get("calls")
        system_hash = str(warmup.get("system_prompt_sha256", ""))
        expected_warm_roles: dict[str, set[str]] = defaultdict(set)
        for role in ("main", "fast"):
            expected_warm_roles[str(role_models.get(role, ""))].add(role)
        observed_warm_roles: dict[str, set[str]] = {}
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                observed_warm_roles[str(call.get("model", ""))] = {
                    str(role) for role in (call.get("roles") or ())
                }
        return bool(
            details.get("model_assignment")
            and role_models.get("main")
            and role_models.get("fast")
            and warmup.get("policy") == "production_system_prompt"
            and warmup.get("performed") is True
            and warmup.get("gate_eligible") is True
            and warmup.get("ok") is True
            and isinstance(calls, list)
            and calls
            and len(calls) == len(expected_warm_roles)
            and all(
                isinstance(call, Mapping) and call.get("ok") is True
                for call in calls
            )
            and observed_warm_roles == expected_warm_roles
            and len(system_hash) == 64
        )

    evaluation_evidence_ok = bool(
        mode != "ollama"
        or (
            evaluation_evidence(candidate_metadata)
            and evaluation_evidence(baseline_metadata)
        )
    )
    warmup_ok = evaluation_evidence_ok
    run_metadata = metadata or {}
    repository_metadata = run_metadata.get("repository")
    config_metadata = run_metadata.get("config")
    provenance_snapshots = run_metadata.get("provenance_snapshots")

    def valid_repository(value: object) -> bool:
        return bool(
            isinstance(value, Mapping)
            and re.fullmatch(
                r"[0-9a-f]{40}",
                str(value.get("revision", "")),
                flags=re.IGNORECASE,
            )
            is not None
            and value.get("dirty") is False
        )

    def valid_config(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        parsed_host = urlparse(str(value.get("host", "")))
        configured_roles = value.get("configured_role_models")
        return bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("contract_sha256", "")),
                flags=re.IGNORECASE,
            )
            is not None
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("llm_options_sha256", "")),
                flags=re.IGNORECASE,
            )
            is not None
            and value.get("include_local_config") is False
            and value.get("backend") == "ollama"
            and (parsed_host.hostname or "").lower()
            in {"localhost", "127.0.0.1", "::1"}
            and parsed_host.username is None
            and parsed_host.password is None
            and isinstance(configured_roles, Mapping)
            and configured_roles.get("main")
            and configured_roles.get("fast")
        )

    before_snapshot = (
        provenance_snapshots.get("before", {})
        if isinstance(provenance_snapshots, Mapping)
        else {}
    )
    after_snapshot = (
        provenance_snapshots.get("after", {})
        if isinstance(provenance_snapshots, Mapping)
        else {}
    )
    repository_before = (
        before_snapshot.get("repository")
        if isinstance(before_snapshot, Mapping)
        else None
    )
    repository_after = (
        after_snapshot.get("repository")
        if isinstance(after_snapshot, Mapping)
        else None
    )
    config_before = (
        before_snapshot.get("config")
        if isinstance(before_snapshot, Mapping)
        else None
    )
    config_after = (
        after_snapshot.get("config")
        if isinstance(after_snapshot, Mapping)
        else None
    )
    repository_clean = bool(
        mode != "ollama"
        or (
            valid_repository(repository_before)
            and valid_repository(repository_after)
        )
    )
    repository_stable = bool(
        mode != "ollama"
        or (
            repository_clean
            and repository_before == repository_after == repository_metadata
        )
    )
    local_config_excluded = bool(
        mode != "ollama"
        or (
            isinstance(config_before, Mapping)
            and isinstance(config_after, Mapping)
            and config_before.get("include_local_config") is False
            and config_after.get("include_local_config") is False
        )
    )
    config_stable = bool(
        mode != "ollama"
        or (
            valid_config(config_before)
            and valid_config(config_after)
            and config_before == config_after == config_metadata
        )
    )
    effective_provenance_ok = bool(
        effective_provenance_ok
        and repository_stable
        and config_stable
        and local_config_excluded
    )
    run_metadata_ok = bool(
        mode != "ollama"
        or (
            repository_stable
            and config_stable
            and local_config_excluded
            and run_metadata.get("warm_policy") == "production"
            and run_metadata.get("execution_order") == ["baseline", "candidate"]
            and str(run_metadata.get("ollama_python_version", ""))
            not in {"", "unavailable", "not_applicable"}
        )
    )
    topology = str((metadata or {}).get("topology", ""))
    candidate_assignment = str(
        (candidate_metadata or {}).get("model_assignment", "")
    )
    baseline_assignment = str(
        (baseline_metadata or {}).get("model_assignment", "")
    )
    expected_assignment = (
        "production_hybrid_fast_override"
        if topology == "production-hybrid"
        else ""
    )
    configured_roles = (
        config_metadata.get("configured_role_models", {})
        if isinstance(config_metadata, Mapping)
        else {}
    )
    configured_main = (
        str(configured_roles.get("main", ""))
        if isinstance(configured_roles, Mapping)
        else ""
    )
    candidate_main = str(candidate_roles.get("main", ""))
    baseline_main = str(baseline_roles.get("main", ""))
    shared_main_contract_equal = bool(
        mode != "ollama"
        or (
            identity_evidence_ok
            and configured_main
            and candidate_contracts.get(configured_main)
            == baseline_contracts.get(configured_main)
            is not None
        )
    )
    adoption_topology_eligible = bool(
        mode != "ollama" or topology == "production-hybrid"
    )
    topology_ok = bool(
        mode != "ollama"
        or (
            evaluation_evidence_ok
            and adoption_topology_eligible
            and expected_assignment
            and candidate_assignment == baseline_assignment == expected_assignment
            and isinstance(candidate_roles, Mapping)
            and isinstance(baseline_roles, Mapping)
            and candidate_roles.get("fast") == candidate_model
            and baseline_roles.get("fast")
            == (baseline_results[0].model if baseline_results else "")
            and candidate_main == baseline_main == configured_main
            and shared_main_contract_equal
        )
    )
    prompt_contract_equal = bool(
        mode != "ollama"
        or (
            evaluation_evidence_ok
            and candidate_metadata["warmup"]["system_prompt_sha256"]
            == baseline_metadata["warmup"]["system_prompt_sha256"]
        )
    )
    candidate_fast_contract = candidate_contracts.get(
        str(candidate_roles.get("fast", ""))
    )
    baseline_fast_contract = baseline_contracts.get(
        str(baseline_roles.get("fast", ""))
    )
    ab_distinct = bool(
        mode != "ollama"
        or (
            identity_evidence_ok
            and isinstance(candidate_roles, Mapping)
            and isinstance(baseline_roles, Mapping)
            and candidate_fast_contract is not None
            and baseline_fast_contract is not None
            and candidate_fast_contract[0] != baseline_fast_contract[0]
        )
    )
    ab_valid = bool(topology_ok and ab_distinct)
    coverage_ok = bool(
        candidate_coverage["ok"]
        and (
            not baseline_required
            or (
                baseline_coverage is not None
                and bool(baseline_coverage["ok"])
            )
        )
    )
    semantic_pass = bool(
        candidate_summary.pass_power_k
        and candidate_model_consistent
        and baseline_model_consistent
        and baseline_valid
        and no_candidate_regression
        and coverage_ok
    )
    report["gate"] = {
        "semantic_pass": semantic_pass,
        "baseline_required": baseline_required,
        "baseline_present": baseline_present,
        "suite_models_consistent": bool(
            candidate_model_consistent and baseline_model_consistent
        ),
        "coverage_ok": coverage_ok,
        "baseline_valid": baseline_valid,
        "no_candidate_regression": no_candidate_regression,
        "warmup_ok": warmup_ok,
        "run_metadata_ok": run_metadata_ok,
        "prompt_contract_equal": prompt_contract_equal,
        "adoption_topology_eligible": adoption_topology_eligible,
        "topology_ok": topology_ok,
        "shared_main_contract_equal": shared_main_contract_equal,
        "ab_distinct": ab_distinct,
        "ab_valid": ab_valid,
        "identity_evidence_ok": identity_evidence_ok,
        "repository_clean": repository_clean,
        "repository_stable": repository_stable,
        "config_stable": config_stable,
        "local_config_excluded": local_config_excluded,
        "provenance_asserted": provenance_asserted,
        "provenance_ok": effective_provenance_ok,
        "diagnostic_override_used": bool(diagnostic_override_used),
        "passed": bool(
            semantic_pass
            and warmup_ok
            and run_metadata_ok
            and prompt_contract_equal
            and ab_valid
            and effective_provenance_ok
        ),
    }
    return report


def write_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
