"""Native-free privacy/metric checks for the bounded local LLM canary."""

import copy

import pytest

from tools import english_llm_benchmark as subject


def _report():
    return {
        "model_id": "minicpm5-1b-q8",
        "status": "ok",
        "model_sha256": subject.MODEL_HASHES["minicpm5-1b-q8"],
        "model_bytes": 1153529216,
        "runtime_version": "0.3.33",
        "template_sha256": "a" * 64,
        "threads": 2,
        "cpu_affinity_enforced": True,
        "gpu_layers": 0,
        "context_tokens": 1024,
        "max_reply_tokens": 64,
        "thinking_requested": False,
        "min_p": 0.0,
        "cold_first_visible_chunk_ms": 100,
        "cold_completion_ms": 200,
        "load_ms": 300,
        "warm_first_visible_chunk_ms": {"p50": 10, "p95": 20},
        "warm_completion_ms": {"p50": 100, "p95": 200},
        "retokenized_completion_tokens_per_second": 10,
        "decoding_mode": "greedy; min_p inactive",
        "canary_scoring": "strict-format-proxy",
        "warm_filesystem_model_load": True,
        "truncated_outputs": 1,
        "empty_outputs": 0,
        "reasoning_marker_outputs": 0,
        "owner_reference_prompts": 8,
        "generations": 36,
        "cpu_seconds": 5,
        "process_peak_rss_bytes": 1024**3,
        "canaries": {
            kind: {"attempts": 3, "passed": 2} for _, kind in subject.CANARIES
        },
    }


@pytest.mark.parametrize(
    "kind,text",
    [
        ("arithmetic", "not42"),
        ("arithmetic", "The answer is not42."),
        ("geography", "not Paris"),
        ("geography", "Paris is not the answer"),
        ("instruction", "ready"),
        ("instruction", "READY!"),
        ("instruction", "READY and secret"),
        ("spelling", "unnecessary"),
    ],
)
def test_negated_or_extra_text_does_not_pass_strict_canary(kind, text):
    assert not subject.canary_passes(kind, text)


@pytest.mark.parametrize(
    "kind,text",
    [
        ("arithmetic", "42"),
        ("arithmetic", "The answer is forty-two."),
        ("geography", "The capital of France is Paris."),
        ("instruction", "READY"),
        ("spelling", "necessary"),
    ],
)
def test_intended_strict_answers_pass(kind, text):
    assert subject.canary_passes(kind, text)


def test_reference_selection_spans_groups_without_retaining_extra_private_fields():
    source = {
        "clips": [
            {
                "text": f"private phrase {group}{index}",
                "group": group,
                "intent": "private intent",
            }
            for group in ("questions", "memory", "corrections")
            for index in range(4)
        ]
    }
    selected = subject.reference_prompts(source)
    assert len(selected) == 8
    assert all(
        any(group in phrase for phrase in selected)
        for group in ("questions", "memory", "corrections")
    )
    assert "private intent" not in " ".join(selected)
    assert len(source["clips"]) == 12


def test_duplicate_or_nonfinite_json_is_rejected(tmp_path):
    for index, payload in enumerate((b'{"clips":[],"clips":[]}', b'{"number":NaN}')):
        path = tmp_path / f"bad{index}.json"
        path.write_bytes(payload)
        with pytest.raises((subject.CanaryError, ValueError)):
            subject._read_json(path)


def test_reading_preserves_input_and_does_not_reject_access_time(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"clips":[]}')
    import os

    os.utime(path, ns=(1, 1))
    value, digest = subject._read_json(path)
    assert value == {"clips": []}
    assert len(digest) == 64
    assert path.read_text() == '{"clips":[]}'


def test_regular_reader_rejects_symlink(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(subject.CanaryError):
        subject._read_json(alias)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(model_sha256="b" * 64),
        lambda r: r.update(runtime_version="private spoken phrase"),
        lambda r: r.update(cpu_seconds=float("nan")),
        lambda r: r.update(threads=True),
        lambda r: r.update(generations=999),
        lambda r: r.update(empty_outputs=37),
        lambda r: r.update(thinking_requested=True),
        lambda r: r.update(warm_filesystem_model_load=False),
        lambda r: r["warm_completion_ms"].update(text="private phrase"),
        lambda r: r["warm_completion_ms"].update(p50=500),
        lambda r: r["canaries"].update(secret={"text": "private phrase"}),
        lambda r: r["canaries"]["instruction"].update(passed=4),
        lambda r: r["canaries"]["instruction"].update(attempts=1),
        lambda r: r["canaries"]["instruction"].update(text="private phrase"),
    ],
)
def test_nested_values_and_bindings_reject_private_or_impossible_results(mutate):
    report = copy.deepcopy(_report())
    mutate(report)
    with pytest.raises(subject.CanaryError):
        subject.validate_report(report, "minicpm5-1b-q8", 2, 3, 8)


def test_valid_report_and_percentile_scope():
    subject.validate_report(_report(), "minicpm5-1b-q8", 2, 3, 8)
    assert subject.percentile([30, 10, 20, 40], 0.5) == 20
    assert subject.percentile([30, 10, 20, 40], 0.95) == 40
    with pytest.raises(subject.CanaryError):
        subject.percentile([float("nan")], 0.5)


def test_native_stdout_stderr_are_suppressed(capfd):
    import os

    with subject._quiet_native():
        os.write(1, b"private phrase\n")
        os.write(2, b"private phrase\n")
    captured = capfd.readouterr()
    assert "private phrase" not in captured.out + captured.err


def test_worker_cleanup_escalates_exact_group_and_waits(monkeypatch):
    events = []

    class Process:
        pid = 1234
        waits = 0

        def wait(self, *, timeout):
            self.waits += 1
            events.append(("wait", timeout))
            if self.waits == 1:
                raise subject.subprocess.TimeoutExpired("fake", timeout)

    monkeypatch.setattr(
        subject.os, "killpg", lambda pid, sig: events.append((pid, sig))
    )
    subject._terminate_worker(Process())
    assert events == [
        (1234, subject.signal.SIGTERM),
        ("wait", 1),
        (1234, subject.signal.SIGKILL),
        ("wait", 5),
    ]
