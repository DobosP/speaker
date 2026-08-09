from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import stat
from types import SimpleNamespace

import pytest

from core.guided_stt_plan import built_in_guided_stt_plan
from tools import guided_stt_pair_attestor as attestor
from tools.tool_route_gate import EXPECTED_ROUTES, ToolRouteGateTotals


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _accuracy(*, errors: int) -> dict[str, object]:
    ref_words, ref_chars = attestor._fixed_reference_totals(16)
    return {
        "clips": 16,
        "nonempty": 16,
        "exact": 16 - errors,
        "wer": round(errors / ref_words, 4),
        "cer": round(errors / ref_chars, 4),
        "word_errors": errors,
        "substitutions": errors,
        "insertions": 0,
        "deletions": 0,
        "ref_words": ref_words,
        "hyp_words": ref_words,
        "char_edits": errors,
        "ref_chars": ref_chars,
        "hyp_chars": ref_chars,
        "keyword_attempts": 0,
        "keyword_hits": 0,
    }


def _route_totals() -> dict[str, object]:
    attempts = dict(attestor._ROUTE_ATTEMPTS)
    return ToolRouteGateTotals(
        annotated_cases=16,
        decisions=16,
        single_decision_cases=16,
        empty_decisions=0,
        expected_positive_cases=11,
        expected_none_cases=5,
        exact_cases=16,
        misses=0,
        wrong_tool=0,
        unexpected_tool=0,
        unexpected_control=0,
        unexpected_action=0,
        multi_decision_cases=0,
        attempts=attempts,
        hits=attempts,
    ).as_dict()


def _evaluation(profile: str, *, errors: int) -> dict[str, object]:
    measured = _accuracy(errors=errors)
    return {
        "clips": 16,
        "decisions": 16,
        "complete": True,
        "selected_sources_attested": True,
        "selected_source_accounting_complete": True,
        "offline_outcomes": {"decoded": 16},
        "verifier_outcomes": (
            {"unavailable": 16}
            if profile == attestor.CONTROL_PROFILE
            else {"consensus": 16}
        ),
        "selected_sources": {"offline": 16},
        "streaming": deepcopy(measured),
        "offline": deepcopy(measured),
        "selected": deepcopy(measured),
        "tool_route_gate": _route_totals(),
    }


def _bundle(role: str) -> attestor._BundleState:
    profile = (
        attestor.CONTROL_PROFILE if role == "control" else attestor.CANDIDATE_PROFILE
    )
    run_dir = Path(f"/private/{role}-bundle")
    return attestor._BundleState(
        source=object(),
        run_dir=run_dir,
        plan_path=run_dir / "plan.json",
        diagnostic_manifest_path=run_dir / "diagnostic.json",
        receipt_sha256=_digest(f"{role}-receipt"),
        plan_sha256=built_in_guided_stt_plan().sha256,
        contract_sha256=_digest(f"{role}-contract"),
        summary_sha256=_digest(f"{role}-summary"),
        diagnostic_manifest_sha256=_digest(f"{role}-manifest"),
        profile=profile,
        profile_sha256=_digest(f"{role}-profile"),
        capture_config_sha256=_digest("capture-config"),
        effective_sherpa_sha256=_digest(f"{role}-sherpa"),
        device_profile="desktop",
        effective_input_gain=1.0,
        case_order_sha256=attestor._fixed_case_order_digest(),
    )


def _corpus(role: str) -> attestor._PreparedState:
    return attestor._PreparedState(
        role=f"{role}-capture",
        labels_path=Path(f"/private/{role}-labels.json"),
        corpus_path=Path(f"/private/{role}-corpus/corpus.json"),
        labels_sha256=_digest(f"{role}-labels"),
        corpus_sha256=_digest(f"{role}-corpus"),
        cases=16,
        audio_bytes=4096,
        receipt_roles=(0, 16),
        loaded_corpus=SimpleNamespace(),  # type: ignore[arg-type]
        labels_identity=(1, 2, 3, 4, 5, 6),
        corpus_identities=((1, 2, 3, 4, 5, 6),),
    )


def _child(
    corpus: attestor._PreparedState,
    control: attestor._BundleState,
    candidate: attestor._BundleState,
    *,
    candidate_errors: int = 1,
) -> dict[str, object]:
    promotable = candidate_errors < 2
    return {
        "ok": promotable,
        "corpus_digest": corpus.corpus_sha256,
        "baseline_config_digest": _digest("control-config"),
        "baseline_model_digest": _digest("control-model"),
        "baseline": _evaluation(attestor.CONTROL_PROFILE, errors=2),
        "baseline_final_stt_profile": attestor.CONTROL_PROFILE,
        "baseline_final_stt_profile_digest": control.profile_sha256,
        "tool_route_profile_digest": attestor._fixed_route_digest(),
        "candidate_config_digest": _digest("candidate-config"),
        "candidate_model_digest": _digest("candidate-model"),
        "candidate": _evaluation(
            attestor.CANDIDATE_PROFILE,
            errors=candidate_errors,
        ),
        "comparison": {
            "wins": 1 if promotable else 0,
            "ties": 15 if promotable else 16,
            "losses": 0,
            "promotable": promotable,
        },
        "candidate_final_stt_profile": attestor.CANDIDATE_PROFILE,
        "candidate_final_stt_profile_digest": candidate.profile_sha256,
    }


def _valid_report() -> dict[str, object]:
    control = _bundle("control")
    candidate = _bundle("candidate")
    corpora = (_corpus("control"), _corpus("candidate"))
    evaluations = tuple(_child(corpus, control, candidate) for corpus in corpora)
    return attestor._report(control, candidate, corpora, evaluations)


def _selected_config(bundle: attestor._BundleState) -> SimpleNamespace:
    return SimpleNamespace(
        final_stt_profile=bundle.profile,
        final_stt_profile_sha256=bundle.profile_sha256,
        final_stt_profile_schema_version=1,
        effective_device=bundle.device_profile,
        effective_input_gain=bundle.effective_input_gain,
        capture_config_sha256=bundle.capture_config_sha256,
        effective_sherpa_sha256=bundle.effective_sherpa_sha256,
    )


def test_validate_report_recomputes_combined_metrics_and_verdict() -> None:
    report = attestor.validate_attestation_report(_valid_report())
    assert report["quality_verdict"] == "candidate_preferred"
    assert report["results"]["combined"]["control"]["clips"] == 32

    nested_extra = deepcopy(report)
    nested_extra["results"]["by_capture"][0]["control"]["transcript"] = "secret"
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(nested_extra)

    changed_total = deepcopy(report)
    changed_total["results"]["combined"]["candidate"]["selected"]["exact"] -= 1
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(changed_total)

    changed_verdict = deepcopy(report)
    changed_verdict["quality_verdict"] = "control_preferred"
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(changed_verdict)


def test_validate_report_binds_fixed_order_and_reference_metrics() -> None:
    report = _valid_report()
    assert attestor._fixed_reference_totals(16) == (112, 488)

    changed_order = deepcopy(report)
    changed_order["contract"]["case_order_sha256"] = _digest("other-order")
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(changed_order)

    impossible_exact = deepcopy(report)
    selected = impossible_exact["results"]["by_capture"][0]["control"]["selected"]
    selected["exact"] = selected["clips"]
    assert selected["word_errors"] > 0
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(impossible_exact)

    forged_denominator = deepcopy(report)
    streaming = forged_denominator["results"]["by_capture"][0]["control"]["streaming"]
    streaming["ref_words"] -= 1
    streaming["wer"] = round(streaming["word_errors"] / streaming["ref_words"], 4)
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.validate_attestation_report(forged_denominator)


def test_write_new_report_is_private_and_no_clobber(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "attestation.json"
    report = _valid_report()

    digest = attestor.write_new_report(output, report)

    metadata = output.lstat()
    raw = output.read_bytes()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert digest == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw) == attestor.validate_attestation_report(report)
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.write_new_report(output, report)
    assert output.read_bytes() == raw


def test_write_new_report_precommit_interrupt_has_no_path_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "attestation.json"
    canary = tmp_path / "foreign-canary"
    canary.write_bytes(b"keep")
    canary.chmod(0o600)

    def interrupt_before_link(*_args, **_kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(attestor, "_commit_report_link", interrupt_before_link)
    monkeypatch.setattr(
        attestor.os,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    with pytest.raises(KeyboardInterrupt):
        attestor.write_new_report(output, _valid_report())
    assert not output.exists()
    assert canary.read_bytes() == b"keep"


def test_write_new_report_rejects_prelink_parent_replacement(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    moved = tmp_path / "moved"
    output = parent / "attestation.json"

    def replace_parent() -> bool:
        parent.rename(moved)
        parent.mkdir(mode=0o700)
        return True

    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.write_new_report(
            output,
            _valid_report(),
            _commit_guard=replace_parent,
        )

    assert not output.exists()
    assert not (moved / output.name).exists()


def test_write_new_report_interrupt_after_terminal_link_keeps_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "attestation.json"
    real_link = attestor.os.link

    def link_then_interrupt(*args, **kwargs) -> None:
        real_link(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(attestor.os, "link", link_then_interrupt)
    digest = attestor.write_new_report(output, _valid_report())
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.is_file()


def test_write_new_report_wrong_name_link_is_not_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "attestation.json"
    wrong = tmp_path / "wrong-name"
    real_link = attestor.os.link

    def wrong_link_then_interrupt(source, _name, **kwargs) -> None:
        real_link(source, wrong.name, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(attestor.os, "link", wrong_link_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        attestor.write_new_report(output, _valid_report())
    assert not output.exists()
    assert wrong.is_file()


def test_write_new_report_interrupt_after_commit_keeps_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "attestation.json"
    real_fsync = attestor.os.fsync
    calls = 0

    def interrupt_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        real_fsync(descriptor)

    monkeypatch.setattr(attestor.os, "fsync", interrupt_directory_fsync)
    digest = attestor.write_new_report(output, _valid_report())
    assert calls == 2
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.is_file()


def test_fixed_child_commands_have_no_public_tuning_surface() -> None:
    bundle = _bundle("control")
    corpus = _corpus("control")
    scratch = Path("/private/scratch")

    prepare = attestor._prepare_command(bundle, scratch, corpus.role)
    evaluate = attestor._evaluation_command(
        corpus,
        bundle.device_profile,
        scratch / "evaluation.json",
    )

    assert prepare[2:4] == ["-m", "tools.prepare_live_stt_corpus"]
    assert prepare[4:] == [
        "--diagnostic-manifest",
        str(bundle.diagnostic_manifest_path),
        "--reference-plan",
        str(bundle.plan_path),
        "--labels-output",
        str(scratch / "control-capture-labels.json"),
        "--output-dir",
        str(scratch / "control-capture-corpus"),
    ]
    assert evaluate[2:4] == ["-m", "tools.recorded_stt_eval"]
    assert evaluate[4:] == [
        "--manifest",
        str(corpus.corpus_path),
        "--device",
        "desktop",
        "--baseline-final-stt-profile",
        attestor.CONTROL_PROFILE,
        "--candidate-final-stt-profile",
        attestor.CANDIDATE_PROFILE,
        "--tool-route-gate",
        "--tool-route-vault-enabled",
        "--tool-route-reminders-enabled",
        "--tool-route-app-alias",
        "obsidian",
        "--output",
        str(scratch / "evaluation.json"),
    ]


def test_run_attestation_publishes_inside_closing_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    scratch = tmp_path / "scratch"
    output = tmp_path / "report.json"
    control = _bundle("control")
    candidate = _bundle("candidate")
    corpora = {
        "control-capture": _corpus("control"),
        "candidate-capture": _corpus("candidate"),
    }
    children = {
        role: _child(corpus, control, candidate) for role, corpus in corpora.items()
    }
    commands: list[list[str]] = []
    bindings: list[str] = []

    monkeypatch.setattr(
        attestor,
        "_load_bundle",
        lambda _path, profile: (
            control if profile == attestor.CONTROL_PROFILE else candidate
        ),
    )
    monkeypatch.setattr(attestor, "_recheck_bundle", lambda _bundle: None)
    monkeypatch.setattr(attestor, "_recheck_prepared", lambda _state, _bundle: None)
    monkeypatch.setattr(
        attestor,
        "_CURRENT_BINDER",
        lambda bundle: bindings.append(bundle.profile) or _selected_config(bundle),
    )
    monkeypatch.setattr(
        attestor,
        "_prepared_state",
        lambda **values: corpora[values["role"]],
    )
    monkeypatch.setattr(
        attestor,
        "_private_json",
        lambda path, **_kwargs: children[path.name.removesuffix("-evaluation.json")],
    )

    def run_child(command: list[str], *, timeout_sec: float) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(attestor, "_CHILD_RUNNER", run_child)

    published = attestor.run_attestation(
        control_bundle="control",
        candidate_bundle="candidate",
        scratch_root=scratch,
        output_path=output,
    )

    assert output.is_file()
    assert published.report_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert published.report["quality_verdict"] == "candidate_preferred"
    assert [command[3] for command in commands] == [
        "tools.prepare_live_stt_corpus",
        "tools.prepare_live_stt_corpus",
        "tools.recorded_stt_eval",
        "tools.recorded_stt_eval",
    ]
    assert [command[command.index("--manifest") + 1] for command in commands[2:]] == [
        str(corpora["control-capture"].corpus_path),
        str(corpora["candidate-capture"].corpus_path),
    ]
    assert bindings[:2] == [attestor.CONTROL_PROFILE, attestor.CANDIDATE_PROFILE]
    assert bindings[-2:] == [attestor.CONTROL_PROFILE, attestor.CANDIDATE_PROFILE]


def test_run_attestation_rebinds_current_config_in_terminal_commit_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    scratch = tmp_path / "scratch"
    output = tmp_path / "report.json"
    control = _bundle("control")
    candidate = _bundle("candidate")
    corpora = {
        "control-capture": _corpus("control"),
        "candidate-capture": _corpus("candidate"),
    }
    children = {
        role: _child(corpus, control, candidate) for role, corpus in corpora.items()
    }
    drifted = False

    monkeypatch.setattr(
        attestor,
        "_load_bundle",
        lambda _path, profile: (
            control if profile == attestor.CONTROL_PROFILE else candidate
        ),
    )
    monkeypatch.setattr(attestor, "_recheck_bundle", lambda _bundle: None)
    monkeypatch.setattr(attestor, "_recheck_prepared", lambda _state, _bundle: None)

    def current_binding(bundle: attestor._BundleState) -> SimpleNamespace:
        selected = _selected_config(bundle)
        if drifted and bundle is candidate:
            selected.capture_config_sha256 = _digest("changed")
        return selected

    monkeypatch.setattr(attestor, "_CURRENT_BINDER", current_binding)
    monkeypatch.setattr(
        attestor,
        "_prepared_state",
        lambda **values: corpora[values["role"]],
    )
    monkeypatch.setattr(
        attestor,
        "_private_json",
        lambda path, **_kwargs: children[path.name.removesuffix("-evaluation.json")],
    )
    monkeypatch.setattr(
        attestor,
        "_CHILD_RUNNER",
        lambda _command, *, timeout_sec: 0,
    )
    real_write = attestor.write_new_report

    def drift_before_staging(*args, **kwargs):
        nonlocal drifted
        assert callable(kwargs.get("_commit_guard"))
        drifted = True
        return real_write(*args, **kwargs)

    monkeypatch.setattr(attestor, "write_new_report", drift_before_staging)

    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.run_attestation(
            control_bundle="control",
            candidate_bundle="candidate",
            scratch_root=scratch,
            output_path=output,
        )

    assert drifted is True
    assert not output.exists()


def test_run_attestation_failure_retains_private_scratch_without_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    scratch = tmp_path / "scratch"
    output = tmp_path / "report.json"
    control = _bundle("control")
    candidate = _bundle("candidate")
    monkeypatch.setattr(
        attestor,
        "_load_bundle",
        lambda _path, profile: (
            control if profile == attestor.CONTROL_PROFILE else candidate
        ),
    )
    monkeypatch.setattr(attestor, "_CURRENT_BINDER", _selected_config)
    monkeypatch.setattr(attestor, "_CHILD_RUNNER", lambda *_args, **_kwargs: 2)

    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.run_attestation(
            control_bundle="control",
            candidate_bundle="candidate",
            scratch_root=scratch,
            output_path=output,
        )

    assert scratch.is_dir()
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
    assert not output.exists()


def test_current_binding_failure_happens_before_child_or_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    scratch = tmp_path / "scratch"
    output = tmp_path / "report.json"
    control = _bundle("control")
    candidate = _bundle("candidate")
    child_called = False
    monkeypatch.setattr(
        attestor,
        "_load_bundle",
        lambda _path, profile: (
            control if profile == attestor.CONTROL_PROFILE else candidate
        ),
    )
    monkeypatch.setattr(
        attestor,
        "_CURRENT_BINDER",
        lambda bundle: SimpleNamespace(
            **{
                **vars(_selected_config(bundle)),
                "capture_config_sha256": _digest("changed"),
            }
        ),
    )

    def child(*_args, **_kwargs) -> int:
        nonlocal child_called
        child_called = True
        return 0

    monkeypatch.setattr(attestor, "_CHILD_RUNNER", child)
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor.run_attestation(
            control_bundle="control",
            candidate_bundle="candidate",
            scratch_root=scratch,
            output_path=output,
        )
    assert child_called is False
    assert not scratch.exists()
    assert not output.exists()


def test_prepared_recheck_rejects_same_content_inode_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _corpus("control")
    monkeypatch.setattr(attestor, "verify_corpus_snapshot", lambda _loaded: None)
    monkeypatch.setattr(
        attestor,
        "_private_file_identity",
        lambda _path: (9, 9, 9, 9, 9, 9),
    )

    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor._recheck_prepared(state, _bundle("control"))


def test_run_child_terminates_process_group_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4321

        def __init__(self) -> None:
            self.waits = 0
            self.alive = True

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            self.alive = False
            return -signal.SIGTERM

        def poll(self) -> int | None:
            return None if self.alive else -signal.SIGTERM

    process = Process()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(attestor.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def killpg(pid: int, signum: int) -> None:
        if signum == 0:
            if process.alive:
                return
            raise ProcessLookupError
        signals.append((pid, signum))

    monkeypatch.setattr(
        attestor.os,
        "killpg",
        killpg,
    )

    with pytest.raises(KeyboardInterrupt):
        attestor._run_child(["child"], timeout_sec=1.0)

    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.waits == 2


def test_run_child_rejects_and_terminates_surviving_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 5432

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    process = Process()
    group_alive = True
    signals: list[int] = []

    def killpg(_pid: int, signum: int) -> None:
        nonlocal group_alive
        if signum == 0:
            if group_alive:
                return
            raise ProcessLookupError
        signals.append(signum)
        group_alive = False

    monkeypatch.setattr(attestor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(attestor.os, "killpg", killpg)
    with pytest.raises(attestor.GuidedSttPairAttestationError):
        attestor._run_child(["child"], timeout_sec=1.0)
    assert signals == [signal.SIGTERM]


def test_run_child_terminates_process_group_on_lifecycle_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 6543

        def __init__(self) -> None:
            self.waits = 0
            self.alive = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise attestor._LifecycleSignal(signal.SIGTERM)
            self.alive = False
            return -signal.SIGTERM

    process = Process()
    signals: list[int] = []

    def killpg(_pid: int, signum: int) -> None:
        if signum == 0:
            if process.alive:
                return
            raise ProcessLookupError
        signals.append(signum)

    monkeypatch.setattr(attestor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(attestor.os, "killpg", killpg)
    with pytest.raises(attestor._LifecycleSignal) as raised:
        attestor._run_child(["child"], timeout_sec=1.0)
    assert raised.value.signum == signal.SIGTERM
    assert signals == [signal.SIGTERM]
    assert process.waits == 2


def test_child_preexec_unblocks_lifecycle_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, set[int]]] = []

    def pthread_sigmask(operation: int, signals: set[int]):
        calls.append((operation, set(signals)))
        return set()

    monkeypatch.setattr(attestor.signal, "pthread_sigmask", pthread_sigmask)
    attestor._unblock_child_lifecycle_signals()
    assert calls == [
        (signal.SIG_UNBLOCK, attestor._child_signal_numbers()),
    ]


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_main_maps_lifecycle_signal_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    signum: int,
) -> None:
    previous = signal.getsignal(signum)

    def interrupted(**_kwargs):
        raise attestor._LifecycleSignal(signum)

    monkeypatch.setattr(attestor, "run_attestation", interrupted)
    assert (
        attestor.main(
            [
                "--control-bundle",
                "/private/control",
                "--candidate-bundle",
                "/private/candidate",
                "--scratch-root",
                "/private/scratch",
                "--output",
                "/private/report.json",
            ]
        )
        == 128 + signum
    )
    assert signal.getsignal(signum) is previous
    assert json.loads(capsys.readouterr().out) == {
        "error": "guided STT pair attestation failed",
        "ok": False,
    }


@pytest.mark.parametrize(
    ("raised", "expected_rc"),
    [
        (attestor.GuidedSttPairAttestationError(), 2),
        (KeyboardInterrupt(), 130),
    ],
)
def test_main_has_detail_free_failure_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raised: BaseException,
    expected_rc: int,
) -> None:
    def fail(**_kwargs):
        raise raised

    monkeypatch.setattr(attestor, "run_attestation", fail)
    rc = attestor.main(
        [
            "--control-bundle",
            "/private/control",
            "--candidate-bundle",
            "/private/candidate",
            "--scratch-root",
            "/private/scratch",
            "--output",
            "/private/report.json",
        ]
    )
    assert rc == expected_rc
    assert json.loads(capsys.readouterr().out) == {
        "error": "guided STT pair attestation failed",
        "ok": False,
    }


@pytest.mark.parametrize("late_error", [BrokenPipeError(), KeyboardInterrupt()])
def test_main_keeps_success_after_terminal_publication(
    monkeypatch: pytest.MonkeyPatch,
    late_error: BaseException,
) -> None:
    report = _valid_report()
    monkeypatch.setattr(
        attestor,
        "run_attestation",
        lambda **_kwargs: attestor.AttestationPublication(
            report=report,
            report_sha256=_digest("published"),
        ),
    )
    monkeypatch.setattr(
        "builtins.print",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(late_error),
    )

    assert (
        attestor.main(
            [
                "--control-bundle",
                "/private/control",
                "--candidate-bundle",
                "/private/candidate",
                "--scratch-root",
                "/private/scratch",
                "--output",
                "/private/report.json",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "late_error",
    [
        KeyboardInterrupt(),
        MemoryError(),
        attestor._LifecycleSignal(signal.SIGTERM),
    ],
)
def test_main_uses_shared_terminal_commit_on_call_return_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    late_error: BaseException,
) -> None:
    report = _valid_report()
    publication = attestor.AttestationPublication(
        report=report,
        report_sha256=_digest("published"),
    )

    def committed_then_fail(**kwargs):
        state = kwargs["_commit_state"]
        state.pending_publication = publication
        state.committed = True
        raise late_error

    monkeypatch.setattr(attestor, "run_attestation", committed_then_fail)
    assert (
        attestor.main(
            [
                "--control-bundle",
                "/private/control",
                "--candidate-bundle",
                "/private/candidate",
                "--scratch-root",
                "/private/scratch",
                "--output",
                "/private/report.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["report_written"] is True


def test_route_attempt_contract_covers_fixed_plan() -> None:
    assert tuple(attestor._ROUTE_ATTEMPTS) == EXPECTED_ROUTES
    assert sum(attestor._ROUTE_ATTEMPTS.values()) == 16
