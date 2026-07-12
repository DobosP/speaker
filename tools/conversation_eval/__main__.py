from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from core.minicpm_identity import MINICPM_Q8_CONTRACT

from .identity import verify_minicpm_identity, verify_ollama_blob_identity
from .provenance import (
    config_metadata,
    failed_identity_records,
    identity_bundle,
    identity_contract,
    identity_snapshot,
    model_identity_record,
    repository_metadata,
)
from .report import build_report, summarize_model, write_report
from .runner import (
    LOCAL_OLLAMA_HEADERS,
    deterministic_models,
    ollama_models,
    run_scenario,
    safe_config,
    warm_models,
)
from .scenarios import SCENARIOS, selected


LOCAL_MODEL = MINICPM_Q8_CONTRACT.alias


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "conversation-eval" / stamp / "report.json"


def _run(models, scenarios, runs: int, config: dict):
    return tuple(
        run_scenario(
            scenario,
            config=config,
            models=models,
            run_index=run_index,
        )
        for run_index in range(1, runs + 1)
        for scenario in scenarios
    )


def _print_summary(label: str, summary) -> None:
    state = "PASS" if summary.pass_power_k else "FAIL"
    print(
        f"{label}: {state} -- {summary.passed_results}/{summary.total_results} "
        f"scenario-runs; pass@1={summary.pass_at_1} "
        f"pass^{summary.runs}={summary.pass_power_k}"
    )
    for scenario_id, row in summary.scenario_reliability.items():
        print(
            f"  {scenario_id}: {row['runs_passed']}/{row['runs']} "
            f"(pass^k={row['pass_power_k']})"
        )


def _repository_metadata() -> dict[str, object]:
    return repository_metadata()


def _config_metadata(config: dict, *, include_local_config: bool) -> dict[str, object]:
    return config_metadata(config, include_local_config=include_local_config)


def _ollama_version() -> str:
    try:
        return importlib_metadata.version("ollama")
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _pair_metadata(models, warmup: dict[str, object]) -> dict[str, object]:
    return {
        "model_assignment": models.model_assignment,
        "role_models": models.role_map(),
        "warmup": warmup,
    }


def _model_identity_record(model: str, config: dict) -> dict[str, object]:
    return model_identity_record(
        model,
        config,
        minicpm_verifier=lambda **kwargs: verify_minicpm_identity(
            **kwargs,
            client_headers=LOCAL_OLLAMA_HEADERS,
        ),
        generic_verifier=lambda name, **kwargs: verify_ollama_blob_identity(
            name,
            **kwargs,
            client_headers=LOCAL_OLLAMA_HEADERS,
        ),
    )


def _identity_contract(record: object) -> tuple[str, str] | None:
    return identity_contract(record)


def _identity_snapshot(role_models: dict[str, str], config: dict) -> dict[str, object]:
    return identity_snapshot(role_models, config, record_fn=_model_identity_record)


def _identity_bundle(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    return identity_bundle(before, after)


def _failed_identity_records(snapshot: dict[str, object]) -> dict[str, object]:
    return failed_identity_records(snapshot)


def _gate_exit_code(gate: dict[str, object]) -> int:
    if not bool(gate.get("provenance_ok")):
        return 2
    return 0 if bool(gate.get("passed")) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the versioned, device-free voice conversation trace gate."
    )
    parser.add_argument("--mode", choices=("deterministic", "ollama"), default="deterministic")
    parser.add_argument("--device", default="desktop_gpu_4090")
    parser.add_argument("--candidate-model", default=LOCAL_MODEL)
    parser.add_argument(
        "--baseline-model",
        default="configured-main",
        help="A/B baseline model; configured-main resolves from the selected device profile",
    )
    parser.add_argument(
        "--topology",
        choices=("production-hybrid", "all-roles"),
        default="production-hybrid",
        help=(
            "production-hybrid replaces only the fast role and keeps the "
            "configured main model; all-roles is an explicit replacement stress test"
        ),
    )
    parser.add_argument(
        "--warm-policy",
        choices=("production", "cold"),
        default="production",
        help="prewarm with the real runtime system prompt, or measure a cold diagnostic",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-local-config", action="store_true")
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help=(
            "continue diagnostically after model identity failure; the report "
            "and process still fail provenance"
        ),
    )
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)

    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(f"{scenario.scenario_id}: {scenario.description}")
        return 0
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    repository_before = _repository_metadata()
    try:
        scenarios = selected(args.scenario)
        config = safe_config(
            device=args.device,
            include_local_config=args.include_local_config,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "ollama" and args.baseline_model == "configured-main":
        args.baseline_model = str(
            (config.get("llm", {}) or {}).get("main_model", "") or ""
        )
    if args.mode == "ollama" and not args.baseline_model.strip():
        parser.error("--mode ollama requires --baseline-model for the A/B gate")

    identities: dict[str, object] = {}
    baseline_results = ()
    baseline_metadata: dict[str, object] = {}
    provenance_ok = True
    diagnostic_override_used = False
    execution_order: list[str] = []
    config_before = _config_metadata(
        config,
        include_local_config=args.include_local_config,
    )
    candidate_identity_before: dict[str, object] | None = None
    if args.mode == "deterministic":
        candidate_models = deterministic_models()
        candidate_metadata = _pair_metadata(
            candidate_models,
            {
                "policy": "not_applicable",
                "performed": False,
                "gate_eligible": True,
                "ok": True,
            },
        )
    else:
        candidate_models = ollama_models(
            config,
            args.candidate_model,
            topology=args.topology,
        )
        print(
            "candidate roles: "
            f"assignment={candidate_models.model_assignment} "
            f"models={candidate_models.role_map()}"
        )
        if args.baseline_model:
            baseline_models = ollama_models(
                config,
                args.baseline_model,
                topology=args.topology,
            )
            baseline_identity_before = _identity_snapshot(
                baseline_models.role_map(),
                config,
            )
            if (
                baseline_identity_before.get("ok") is not True
                and not args.allow_unverified_model
            ):
                parser.exit(
                    2,
                    "Baseline identity verification failed: "
                    f"{_failed_identity_records(baseline_identity_before)}\n",
                )
            baseline_warmup = (
                warm_models(config, baseline_models)
                if args.warm_policy == "production"
                else {
                    "policy": "cold",
                    "performed": False,
                    "gate_eligible": False,
                    "ok": True,
                    "calls": [],
                }
            )
            baseline_metadata = _pair_metadata(
                baseline_models,
                baseline_warmup,
            )
            print(
                "baseline roles: "
                f"assignment={baseline_models.model_assignment} "
                f"models={baseline_models.role_map()}"
            )
            execution_order.append("baseline")
            baseline_results = _run(
                baseline_models,
                scenarios,
                args.runs,
                config,
            )
            baseline_identity_after = _identity_snapshot(
                baseline_models.role_map(),
                config,
            )
            identities["baseline"] = _identity_bundle(
                baseline_identity_before,
                baseline_identity_after,
            )
        candidate_identity_before = _identity_snapshot(
            candidate_models.role_map(),
            config,
        )
        if (
            candidate_identity_before.get("ok") is not True
            and not args.allow_unverified_model
        ):
            parser.exit(
                2,
                "Candidate identity verification failed: "
                f"{_failed_identity_records(candidate_identity_before)}\n",
            )
        candidate_warmup = (
            warm_models(config, candidate_models)
            if args.warm_policy == "production"
            else {
                "policy": "cold",
                "performed": False,
                "gate_eligible": False,
                "ok": True,
                "calls": [],
            }
        )
        candidate_metadata = _pair_metadata(candidate_models, candidate_warmup)

    execution_order.append("candidate")
    candidate_results = _run(candidate_models, scenarios, args.runs, config)
    if args.mode == "ollama":
        assert candidate_identity_before is not None
        candidate_identity_after = _identity_snapshot(
            candidate_models.role_map(),
            config,
        )
        identities["candidate"] = _identity_bundle(
            candidate_identity_before,
            candidate_identity_after,
        )
        provenance_ok = bool(
            identities
            and all(
                isinstance(bundle, dict) and bundle.get("ok") is True
                for bundle in identities.values()
            )
        )
        diagnostic_override_used = bool(
            args.allow_unverified_model and not provenance_ok
        )
    candidate_summary = summarize_model(candidate_models.label, candidate_results)
    _print_summary("candidate", candidate_summary)
    if baseline_results:
        baseline_summary = summarize_model(args.baseline_model, baseline_results)
        _print_summary("baseline", baseline_summary)

    try:
        config_after_value = safe_config(
            device=args.device,
            include_local_config=args.include_local_config,
        )
        config_after = _config_metadata(
            config_after_value,
            include_local_config=args.include_local_config,
        )
    except (OSError, ValueError) as exc:
        config_after = {"error": f"{type(exc).__name__}: {exc}"}
    repository_after = _repository_metadata()

    report = build_report(
        mode=args.mode,
        device=args.device,
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        identities=identities,
        metadata={
            "repository": repository_after,
            "config": config_after,
            "provenance_snapshots": {
                "before": {
                    "repository": repository_before,
                    "config": config_before,
                },
                "after": {
                    "repository": repository_after,
                    "config": config_after,
                },
            },
            "topology": (
                args.topology if args.mode == "ollama" else "deterministic"
            ),
            "warm_policy": (
                args.warm_policy if args.mode == "ollama" else "not_applicable"
            ),
            "execution_order": execution_order,
            "ollama_python_version": (
                _ollama_version() if args.mode == "ollama" else "not_applicable"
            ),
        },
        candidate_metadata=candidate_metadata,
        baseline_metadata=baseline_metadata,
        provenance_ok=provenance_ok,
        diagnostic_override_used=diagnostic_override_used,
    )
    output = write_report(args.output or _default_output(), report)
    gate = report["gate"]
    print(
        f"gate: {'PASS' if gate['passed'] else 'FAIL'} -- "
        f"semantics={gate['semantic_pass']} coverage={gate['coverage_ok']} "
        f"ab={gate['ab_valid']} "
        f"provenance={gate['provenance_ok']} "
        f"warmup={gate['warmup_ok']}"
    )
    print(f"report: {output}")
    return _gate_exit_code(gate)


if __name__ == "__main__":
    raise SystemExit(main())
