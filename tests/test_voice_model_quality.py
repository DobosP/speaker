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
    monkeypatch.setattr(quality, "public_factory_client", lambda host, device: (FactoryModel(), "fixed spoken factory system", options))
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
        assert choices == ("ACT", "INGEST", "UNSURE")
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
