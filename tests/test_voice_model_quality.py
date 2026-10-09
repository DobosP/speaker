"""Public benchmark scoring/coverage/provenance; no models or audio."""
from __future__ import annotations

import json

import pytest

from tools import voice_model_quality as quality


@pytest.mark.parametrize('wrong', ['4.2', '-4', '14', '4 4', 'the answer is 4'])
def test_exact_arithmetic_never_accepts_substrings_or_lost_signs(wrong):
    assert not quality.score(quality.CASES[0], wrong, 'public style')['exact']


@pytest.mark.parametrize('right', ['4', '4.', 'Four!'])
def test_expected_exact_spoken_and_numeric_forms(right):
    assert quality.score(quality.CASES[0], right, 'public style')['exact']


def test_long_instruction_overlap_is_diagnostic_not_generic_word_match():
    system = 'Answer the latest user request directly and keep simple answers brief without inventing facts.'
    assert quality.recites_instruction(system, system)
    assert not quality.recites_instruction('The answer is four.', system)


def test_confirmation_questions_are_distinct_from_prompt_selection_cases():
    first = {case.prompt for case in quality.CASES}
    confirmation = {case.prompt for case in quality.CONFIRMATION_CASES}
    assert len(first) == 16
    assert len(confirmation) == 8
    assert not first.intersection(confirmation)
    assert all(case.split == 'confirmation' for case in quality.CONFIRMATION_CASES)


@pytest.mark.parametrize('host', ['https://127.0.0.1:11435', 'http://192.168.1.2:11435',
                                   'http://example.com:11435', 'http://user:pass@127.0.0.1:11435',
                                   'http://127.0.0.1:11435/path', 'http://127.0.0.1:11435?q=secret',
                                   'http://127.0.0.1'])
def test_nonlocal_or_ambiguous_endpoint_is_refused(host):
    with pytest.raises(ValueError):
        quality.local_host(host)


def test_loopback_requires_explicit_port():
    assert quality.local_host('http://127.0.0.1:11435') == 'http://127.0.0.1:11435'


def fake_models(monkeypatch):
    calls = []
    monkeypatch.setattr(quality, 'identity', lambda *args: ('a' * 64, 'b' * 64))
    by_prompt = {case.prompt: case.expected[0] for case in quality.CASES + quality.CONFIRMATION_CASES}
    class Model:
        def __init__(self, *args, **kwargs):
            assert kwargs['options']['num_predict'] == 128
            assert kwargs['options']['num_ctx'] == 4096
        def generate(self, prompt, *, system, history=None):
            calls.append(prompt)
            return by_prompt[prompt]
        def stream(self, prompt, **kwargs):
            yield self.generate(prompt, **kwargs)
    monkeypatch.setattr(quality, 'OllamaLLM', Model)
    monkeypatch.setattr(quality, 'systems', lambda: {'current': 'public current', 'minimal_voice': 'public candidate'})
    return calls


def test_report_has_complete_selected_case_coverage_and_no_output_text(monkeypatch):
    calls = fake_models(monkeypatch)
    report = quality.run(quality.MODELS[0], 'http://127.0.0.1:11435',
                         confirmation=True, conditions=('current', 'minimal_voice'))
    assert len(calls) == report['calls'] == 16
    assert len(report['cells']) == 2
    assert all(row['calls'] == row['exact'] == 8 for row in report['cells'])
    encoded = json.dumps(report)
    assert 'eleven plus five' not in encoded
    assert 'public_output' not in encoded
    assert 'prompt' not in report
    assert len(report['source_sha256']) == len(report['persona_source_sha256']) == 64


def test_model_change_refuses_result_instead_of_borrowing_identity(monkeypatch):
    fake_models(monkeypatch)
    identities = iter([('a' * 64, 'b' * 64), ('c' * 64, 'b' * 64)])
    monkeypatch.setattr(quality, 'identity', lambda *args: next(identities))
    with pytest.raises(ValueError, match='model_identity_changed'):
        quality.run(quality.MODELS[0], 'http://127.0.0.1:11435', conditions=('current',))


def test_existing_output_is_not_replaced_or_native_called(monkeypatch, tmp_path, capsys):
    output = tmp_path / 'report.json'
    output.write_text('preserve')
    monkeypatch.setattr(quality, 'run', lambda *args, **kwargs: pytest.fail('must not start model'))
    assert quality.main(['--host', 'http://127.0.0.1:11435', '--model', quality.MODELS[0],
                         '--output', str(output)]) == 2
    assert output.read_text() == 'preserve'
    assert 'output_exists' in capsys.readouterr().err


def test_factory_holdout_is_disjoint_and_has_complete_exact_references():
    earlier = {case.prompt for case in quality.CASES + quality.CONFIRMATION_CASES}
    assert len(quality.FACTORY_HOLDOUT_CASES) == 8
    assert not earlier.intersection(case.prompt for case in quality.FACTORY_HOLDOUT_CASES)
    assert all(case.expected and case.split == "factory_holdout" for case in quality.FACTORY_HOLDOUT_CASES)


def test_cli_host_credentials_are_not_exposed_and_no_model_starts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(quality, "run", lambda *a, **k: pytest.fail("must reject endpoint first"))
    assert quality.main(["--host", "http://u:PASSWORD_CANARY@127.0.0.1:11435", "--model", quality.MODELS[0],
                         "--output", str(tmp_path / "result.json")]) == 2
    output = capsys.readouterr()
    assert "PASSWORD_CANARY" not in output.out + output.err


def test_factory_mode_uses_actual_returned_budget_and_fixed_spoken_system(monkeypatch):
    fake_models(monkeypatch)
    options = {"num_ctx": 8192, "num_predict": 512, "num_thread": 2}
    class FactoryModel:
        def generate(self, prompt, **kwargs):
            assert kwargs["system"] == "fixed spoken factory system"
            return "Iris" if "name" in prompt else "4"
        def stream(self, prompt, **kwargs):
            yield self.generate(prompt, **kwargs)
    monkeypatch.setattr(quality, "public_factory_client", lambda host, device: (FactoryModel(), "fixed spoken factory system", options, quality.LLMAddressingClassifier(FactoryModel(), prompt_profile="qwen2.5-1.5b")))
    monkeypatch.setattr(quality, "systems", lambda: {"spoken": "component system"})
    report = quality.run(quality.MODELS[2], "http://127.0.0.1:11435", factory_profile="desktop_gpu_4090", conditions=("spoken",))
    assert report["factory_profile"] and report["options"] == options
    assert all(row["condition"] == "spoken" for row in report["cells"])


def test_decision_checks_use_exact_fixed_cases_and_unchanged_deadline(monkeypatch):
    from core.llm_decision import decision_request
    calls = []
    class Client:
        def generate(self, prompt, **kwargs):
            request = decision_request.get()
            assert request.max_tokens == 16 and request.timeout_sec == 3.0
            return '"ACT"'
    expected = dict(quality.DECISION_CASES)
    def collect(client, prompt, *, system, choices):
        assert choices == quality.DECISION_CHOICES
        # Actual classifier wraps the fixed text, without accepting caller data.
        text = next(text for text in expected if text in prompt)
        calls.append(text)
        return expected[text]
    monkeypatch.setattr(quality, "collect_llm_decision", collect)
    result, warm = quality.resident_decisions(Client())
    assert result["calls"] == result["correct"] == result["available"] == 8
    assert result["false_act"] == 0 and len(calls) == 8
    assert result["timeout_seconds"] == 3 and result["max_tokens"] == 16
    assert warm >= 0


def test_process_resource_sampling_has_only_aggregate_scalars(tmp_path, monkeypatch):
    real_path = quality.Path
    def mapped(path):
        return tmp_path / str(path).lstrip("/")
    monkeypatch.setattr(quality, "Path", mapped)
    # Child belongs to a worker thread, absent from the main thread's children.
    for pid, tid, children in ((10, 10, ""), (10, 11, "20"), (20, 20, "")):
        task = tmp_path / f"proc/{pid}/task/{tid}"
        task.mkdir(parents=True)
        (task / "children").write_text(children)
        (tmp_path / f"proc/{pid}/status").write_text("VmRSS:\t100 kB\n")
    result = quality.process_tree_sample(10)
    assert result == {"rss_bytes": 200 * 1024, "processes": 2, "threads": 3, "read_failures": 0}
    assert all(type(value) is int for value in result.values())
    monkeypatch.setattr(quality, "Path", real_path)


def test_interleaved_check_never_prewarms_decision_prefix_between_answer_calls(monkeypatch):
    fake_models(monkeypatch)
    sequence = []
    class Client:
        def generate(self, prompt, **kwargs):
            sequence.append("answer_warm")
            return "READY"
        def stream(self, prompt, **kwargs):
            sequence.append("answer")
            yield "4"
    def collect(*args, **kwargs):
        sequence.append("decision")
        return None
    monkeypatch.setattr(quality, "public_factory_client", lambda *a: (Client(), "spoken", {"num_thread": 2}, quality.LLMAddressingClassifier(Client(), prompt_profile="qwen2.5-1.5b")))
    monkeypatch.setattr(quality, "collect_llm_decision", collect)
    result = quality.run(quality.MODELS[2], "http://127.0.0.1:11435",
                         factory_profile="cpu_laptop", interleave_checks=True)
    assert sequence == ["answer_warm"] + ["answer", "decision"] * 4
    assert result["calls"] == 4
    assert result["interleaved_decision"]["calls"] == 4
    assert result["interleaved_decision"]["available"] == 0
    assert result["decision_warm_seconds"] is None


def test_adversarial_matrix_keeps_full_shortcuts_separate_from_forced_native_semantics(monkeypatch):
    from core.llm_decision import current_decision_request
    calls = []
    class Model:
        def stream(self, prompt, **kwargs):
            assert current_decision_request() is not None
            calls.append(prompt)
            yield '"INGEST"'
    model = Model()
    gate = quality.LLMAddressingClassifier(model, prompt_profile="qwen2.5-1.5b")
    report = quality.adversarial_matrix(model, gate)
    groups = {row["kind"]: row for row in report["groups"]}
    assert groups["positive"]["shortcut"] == 2
    assert groups["negative"]["shortcut"] == 0
    assert groups["negative"]["forced_available"] == groups["negative"]["forced_correct"] == 16
    assert groups["ambiguous"]["strict_scored_calls"] == 0
    assert len(calls) == 32 * 2 - 2


def test_mixed_context_is_fixed_public_and_never_trimmed_to_make_timing_pass():
    assert len(quality.PUBLIC_RECENT) == 4
    assert quality.MIXED_DECISION_CASES[2] == ("Paris.", "ACT", quality.PUBLIC_RECENT)
    assert quality.CONTEXT_CASES[0][3] is quality.PUBLIC_RECENT


def test_startup_direct_prefill_is_diagnostic_only_bounded_local_and_restores_context():
    from core.llm import capability_context
    from core.llm_decision import current_decision_request
    from always_on_agent.models import CLOUD_EGRESS_SCOPE_CONTEXT_KEY, CloudEgressScope
    class Client:
        def stream(self, prompt, **kwargs):
            request = current_decision_request()
            context = capability_context.get()
            assert request.max_tokens == 16
            assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
            assert 19 < context["cancel_event"].deadline - quality.time.monotonic() <= 20
            yield '"INGEST"'
    client = Client()
    token = capability_context.set({"marker": "original"})
    try:
        result = quality.diagnostic_startup_prefill(client, quality.LLMAddressingClassifier(client))
        assert result["available"] and result["startup_timeout_seconds"] == 20
        assert capability_context.get() == {"marker": "original"}
        assert current_decision_request() is None
    finally:
        capability_context.reset(token)


def fake_source_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "_SOURCE_ROOT", tmp_path)
    for relative in quality._SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixed public source fixture " + relative).encode())
    return tmp_path


def test_prospective_manifest_is_exact_bounded_and_contains_only_public_names_hashes_counts(monkeypatch, tmp_path):
    import hashlib
    root = fake_source_tree(monkeypatch, tmp_path)
    result = quality.source_manifest()
    assert result["file_count"] == len(quality._SOURCE_FILES) == 15
    assert result["total_bytes"] <= quality._SOURCE_MAX_TOTAL_BYTES
    assert [row["file"] for row in result["files"]] == list(quality._SOURCE_FILES)
    assert all(set(row) == {"file", "bytes", "sha256"} for row in result["files"])
    assert all(type(row["bytes"]) is int and 0 < row["bytes"] <= quality._SOURCE_MAX_FILE_BYTES for row in result["files"])
    payload = {key: value for key, value in result.items() if key != "sha256"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert result["sha256"] == expected
    assert result == quality.source_manifest()
    encoded = json.dumps(result)
    assert str(root) not in encoded and "fixed public source fixture" not in encoded
    assert "not_full_import_or_audio_closure" in result["scope"]


@pytest.mark.parametrize("relative", quality._SOURCE_FILES)
def test_any_bound_behavior_source_mutation_during_run_refuses_public_result(monkeypatch, tmp_path, relative):
    root = fake_source_tree(monkeypatch, tmp_path)
    fake_models(monkeypatch)
    class ChangedModel:
        def __init__(self, *args, **kwargs):
            self.changed = False
        def stream(self, prompt, **kwargs):
            if not self.changed:
                (root / relative).write_bytes(b"changed source; never expose this content")
                self.changed = True
            yield "4"
    monkeypatch.setattr(quality, "OllamaLLM", ChangedModel)
    with pytest.raises(ValueError, match="^evaluation_source_changed$"):
        quality.run(quality.MODELS[0], "http://127.0.0.1:11435", conditions=("current",))


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory", "oversized", "total_oversized"])
def test_invalid_before_snapshot_refuses_before_model_identity_or_native_construction(monkeypatch, tmp_path, kind):
    root = fake_source_tree(monkeypatch, tmp_path)
    path = root / "core/llm.py"
    if kind == "missing":
        path.unlink()
    elif kind == "symlink":
        target = tmp_path / "external-secret-canary"
        target.write_bytes(b"SECRET_CANARY_DO_NOT_EXPOSE")
        path.unlink()
        path.symlink_to(target)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "oversized":
        monkeypatch.setattr(quality, "_SOURCE_MAX_FILE_BYTES", 8)
    else:
        monkeypatch.setattr(quality, "_SOURCE_MAX_TOTAL_BYTES", 8)
    monkeypatch.setattr(quality, "identity", lambda *a: pytest.fail("source validation must come first"))
    with pytest.raises(ValueError, match="^evaluation_source_unavailable$"):
        quality.run(quality.MODELS[0], "http://127.0.0.1:11435", conditions=("current",))


def test_source_disappearing_after_measurement_fails_closed_without_filesystem_details(monkeypatch, tmp_path):
    root = fake_source_tree(monkeypatch, tmp_path)
    fake_models(monkeypatch)
    class Model:
        def __init__(self, *a, **k):
            pass
        def stream(self, prompt, **kwargs):
            (root / "core/addressing.py").unlink(missing_ok=True)
            yield "4"
    monkeypatch.setattr(quality, "OllamaLLM", Model)
    with pytest.raises(ValueError, match="^evaluation_source_changed$"):
        quality.run(quality.MODELS[0], "http://127.0.0.1:11435", conditions=("current",))


def test_success_receipt_binds_pre_and_post_seam_and_keeps_legacy_hash_fields(monkeypatch, tmp_path):
    fake_source_tree(monkeypatch, tmp_path)
    fake_models(monkeypatch)
    result = quality.run(quality.MODELS[0], "http://127.0.0.1:11435", conditions=("current",))
    manifest = result["source_manifest"]
    hashes = {row["file"]: row["sha256"] for row in manifest["files"]}
    assert result["source_manifest_after_sha256"] == manifest["sha256"]
    assert result["source_sha256"] == hashes["tools/voice_model_quality.py"]
    assert result["persona_source_sha256"] == hashes["core/persona.py"]


def test_cli_source_refusal_is_coarse_and_never_writes_a_report(monkeypatch, tmp_path, capsys):
    root = fake_source_tree(monkeypatch, tmp_path)
    (root / "core/llm_factory.py").unlink()
    output = tmp_path / "must-not-publish.json"
    assert quality.main(["--host", "http://127.0.0.1:11435", "--model", quality.MODELS[0],
                         "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "voice_model_quality_refused:evaluation_source_unavailable"
    assert str(tmp_path) not in captured.out + captured.err
    assert not output.exists()
