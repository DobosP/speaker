from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from tools.setup_minicpm import LOCAL_MODEL

from .identity import verify_minicpm_identity
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


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _repository_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
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


def _config_metadata(config: dict, *, include_local_config: bool) -> dict[str, object]:
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
    contract = {
        "input_gate": config.get("input_gate"),
        "cleanup": config.get("cleanup"),
        "capability_router": config.get("capability_router"),
        "agent": config.get("agent"),
        "recent_context": config.get("recent_context"),
        "llm": {
            key: llm.get(key)
            for key in (
                "backend",
                "main_model",
                "fast_model",
                "router_model",
                "think",
                "options",
                "keep_alive",
            )
        },
    }
    return {
        "include_local_config": include_local_config,
        "contract_sha256": _json_sha256(contract),
        "llm_options_sha256": _json_sha256(llm.get("options")),
        "backend": str(llm.get("backend", "ollama")),
        "host": host_label,
        "keep_alive": llm.get("keep_alive"),
        "configured_role_models": {
            "main": str(llm.get("main_model", "")),
            "fast": str(llm.get("fast_model", "")),
        },
    }


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
            "continue diagnostically after MiniCPM identity failure; the report "
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
        identity_cache: dict[str, object] = {}

        def identity_record(model: str) -> dict[str, object]:
            if model != LOCAL_MODEL:
                return {
                    "model": model,
                    "verification": "name_only_no_digest_contract",
                    "required": False,
                    "ok": None,
                }
            if model not in identity_cache:
                identity_cache[model] = verify_minicpm_identity(
                    host=str((config.get("llm", {}) or {}).get("host", "") or "")
                    or None,
                    client_headers=LOCAL_OLLAMA_HEADERS,
                )
            identity = identity_cache[model]
            return {
                "model": model,
                "verification": "minicpm_q8_blob_template_parameters",
                "required": True,
                **identity.as_dict(),
            }

        identities["candidate"] = identity_record(args.candidate_model)
        if args.baseline_model:
            identities["baseline"] = identity_record(args.baseline_model)
        provenance_ok = all(
            not bool(record.get("required")) or bool(record.get("ok"))
            for record in identities.values()
            if isinstance(record, dict)
        )
        diagnostic_override_used = bool(
            args.allow_unverified_model and not provenance_ok
        )
        if not provenance_ok and not args.allow_unverified_model:
            failed = {
                label: record
                for label, record in identities.items()
                if isinstance(record, dict)
                and record.get("required")
                and not record.get("ok")
            }
            parser.exit(
                2,
                f"MiniCPM identity verification failed: {failed}\n",
            )

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
    candidate_summary = summarize_model(candidate_models.label, candidate_results)
    _print_summary("candidate", candidate_summary)
    if baseline_results:
        baseline_summary = summarize_model(args.baseline_model, baseline_results)
        _print_summary("baseline", baseline_summary)

    report = build_report(
        mode=args.mode,
        device=args.device,
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        identities=identities,
        metadata={
            "repository": _repository_metadata(),
            "config": _config_metadata(
                config,
                include_local_config=args.include_local_config,
            ),
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
