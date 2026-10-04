"""Privacy, CPU configuration, immutable input and aggregate metrics tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from tools import english_asr_benchmark as subject


def _wav(path, *, rate=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b"\0" * (rate * channels * width // 10))


def _inputs(tmp_path, *, roles=("round_trip", "command", "barge")):
    root = tmp_path / "recordings"
    root.mkdir()
    rows = []
    for index, role in enumerate(roles):
        name = f"private-name-{index}.wav"
        _wav(root / name)
        rows.append(
            {
                "file": name,
                "text": "private reference words",
                "role": role,
                "group": "private group",
                "intent": "private intent",
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"clips": rows}))
    model_root = tmp_path / "model"
    model_root.mkdir()
    model, tokens = model_root / "model.onnx", model_root / "tokens.txt"
    model.write_bytes(b"fake model")
    tokens.write_bytes(b"fake tokens")
    return manifest, subject.Model("sensevoice-en", {"model": model, "tokens": tokens})


def test_corpus_and_report_never_render_references_paths_or_filenames(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    assert "private" not in repr(corpus)
    assert "private" not in repr(corpus.clips[0])
    outputs = iter(["private reference words", "", "different words"] * 3)
    result = subject.benchmark_cell(
        model,
        corpus,
        3,
        2,
        decoder_factory=lambda _m, _t: lambda _samples: next(outputs),
    )
    serialized = json.dumps(result)
    for private in (
        "private reference words",
        "different words",
        "private-name",
        "private intent",
        str(tmp_path),
    ):
        assert private not in serialized
    assert result["clips"] == 3
    assert result["calls"] == 9
    assert result["accuracy"]["exact"] == 1
    assert result["accuracy"]["empty"] == 1
    assert result["accuracy"]["reference_words"] == 9
    assert result["by_role"]["command"]["empty"] == 1
    assert result["by_role"]["barge"]["word_errors"] > 0
    assert result["complete_corpus_coverage"] is True
    assert result["decode_rtf"] == pytest.approx(result["decode_seconds"] / 0.9)
    assert result["peak_rss_bytes"] > 0
    assert result["model_load_ms"] >= 0
    assert result["first_call_ms"] >= 0
    assert result["warm_p50_ms"] is not None
    assert result["warm_p95_ms"] is not None
    result["cpu_affinity"] = {"applied": False, "logical_cpus": None}
    subject._validate_cell_result(result, model.id, 3, 3)
    forged = copy.deepcopy(result)
    forged["accuracy"]["private reference words"] = 1
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(forged, model.id, 3, 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("file", "../outside.wav"),
        ("file", "/absolute.wav"),
        ("text", ""),
        ("role", "private label"),
    ],
)
def test_corpus_rejects_traversal_unlabelled_and_unknown_roles(tmp_path, field, value):
    manifest, _model = _inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["clips"][0][field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(subject.BenchmarkError):
        subject.load_corpus(manifest)


def test_corpus_rejects_symlink_and_detects_post_load_mutation(tmp_path):
    manifest, _model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    original = corpus.clips[0].path.read_bytes()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(original)
    corpus.clips[0].path.unlink()
    corpus.clips[0].path.symlink_to(outside)
    with pytest.raises(subject.BenchmarkError):
        subject.verify_corpus(corpus)
    corpus.clips[0].path.unlink()
    corpus.clips[0].path.write_bytes(original)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(subject.BenchmarkError):
        subject.verify_corpus(corpus)


@pytest.mark.parametrize(
    "rate,channels,width", [(8000, 1, 2), (16000, 2, 2), (16000, 1, 1)]
)
def test_corpus_only_accepts_declared_mono_pcm16_16k(tmp_path, rate, channels, width):
    manifest, _model = _inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    _wav(
        manifest.parent / payload["clips"][0]["file"],
        rate=rate,
        channels=channels,
        width=width,
    )
    with pytest.raises(subject.BenchmarkError):
        subject.load_corpus(manifest)


def test_legacy_source_root_and_hashes_are_bound_without_relabelling(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    clips = source_root / "audio"
    clips.mkdir()
    _wav(clips / "private-id.wav")
    payload = {
        "clip_dir": "audio",
        "clips": [
            {
                "id": "private-id",
                "expected_text": "private reference",
                "sha256": hashlib.sha256(
                    (clips / "private-id.wav").read_bytes()
                ).hexdigest(),
            }
        ],
        "barge": [{"unrelated": "private"}],
        "notes": "private notes",
    }
    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    loaded = subject.load_corpus(manifest, source_root)
    assert loaded.kind == "recorded-legacy"
    assert len(loaded.clips) == 1
    assert loaded.clips[0].reference == "private reference"
    payload["clips"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(subject.BenchmarkError):
        subject.load_corpus(manifest, source_root)


def test_mutation_during_decode_fails_and_owned_decoder_closes(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    closed = []

    class Decoder:
        def __call__(self, _samples):
            model.artifacts["model"].write_bytes(b"changed model")
            return "private reference words"

        def close(self):
            closed.append(True)

    with pytest.raises(subject.BenchmarkError):
        subject.benchmark_cell(
            model, corpus, 1, 2, decoder_factory=lambda _m, _t: Decoder()
        )
    assert closed == [True]


def test_sherpa_cpu_configuration_and_declared_streaming_flush_padding(
    monkeypatch, tmp_path
):
    manifest, sense = _inputs(tmp_path)
    samples = np.ones(1600, dtype="float32")
    calls, fed = [], []

    class Stream:
        result = SimpleNamespace(text="result")

        def accept_waveform(self, rate, waveform):
            fed.append((rate, len(waveform), bool(np.any(waveform))))

        def input_finished(self):
            fed.append("finished")

    class Recognizer:
        def create_stream(self):
            return Stream()

        def decode_stream(self, _stream):
            pass

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return "result"

    def create(**kwargs):
        calls.append(kwargs)
        return Recognizer()

    module = SimpleNamespace(
        OfflineRecognizer=SimpleNamespace(
            from_sense_voice=create, from_transducer=create
        ),
        OnlineRecognizer=SimpleNamespace(from_transducer=create),
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    assert subject._decoder(sense, 2)(samples) == "result"
    for identifier in (
        "zipformer-en",
        "zipformer-en-int8",
        "parakeet-unified-en",
        "parakeet-tdt-v3",
    ):
        artifacts = {
            key: Path(key) for key in ("encoder", "decoder", "joiner", "tokens")
        }
        assert (
            subject._decoder(subject.Model(identifier, artifacts), 2)(samples)
            == "result"
        )
    assert all(call["provider"] == "cpu" and call["num_threads"] == 2 for call in calls)
    assert calls[0]["language"] == "en" and calls[0]["use_itn"] is True
    assert calls[-1]["model_type"] == "nemo_transducer"
    assert fed == [
        (16000, 1600, True),
        (16000, 1600, True),
        (16000, 10560, False),
        "finished",
        (16000, 1600, True),
        (16000, 10560, False),
        "finished",
        (16000, 1600, True),
        (16000, 1600, True),
    ]
    np.testing.assert_array_equal(samples, np.ones(1600, dtype="float32"))


def test_faster_whisper_uses_local_cpu_int8_and_explicit_english(monkeypatch):
    calls = []

    class Whisper:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def transcribe(self, _samples, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text="decoded")]), None

    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper)
    )
    decode = subject._decoder(
        subject.Model("faster-whisper-small", {"directory": Path("local-model")}), 2
    )
    assert decode(np.ones(1600)) == "decoded"
    assert calls[0][1] == {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 2,
        "num_workers": 1,
        "local_files_only": True,
    }
    assert calls[1]["language"] == "en"
    assert calls[1]["condition_on_previous_text"] is False
    assert calls[1]["vad_filter"] is False


def test_timeout_reaps_exact_process_group_and_sanitizes_environment(
    tmp_path, monkeypatch
):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    observed, killed = [], []

    class Process:
        pid = 12345

        def __init__(self):
            self.running = True

        def poll(self):
            return None if self.running else 0

        def wait(self, timeout):
            if timeout == 5:
                raise subprocess.TimeoutExpired("private process", timeout)
            self.running = False
            return 0

    def popen(argv, **kwargs):
        observed.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    monkeypatch.setattr(
        subject.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    monkeypatch.setenv("PRIVATE_CREDENTIAL", "private credential")
    result = subject.run_cell(model, corpus, scratch, 3, 2, 5)
    assert result["status"] == "worker_timeout"
    assert result["complete_corpus_coverage"] is False
    assert killed and killed[0][0] == 12345
    argv, kwargs = observed[0]
    assert "-I" in argv and "-B" in argv
    assert kwargs["start_new_session"] is True
    assert (
        kwargs["stdout"] == subprocess.DEVNULL
        and kwargs["stderr"] == subprocess.DEVNULL
    )
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert "PRIVATE_CREDENTIAL" not in kwargs["env"]
    assert list(scratch.iterdir()) == []


def test_cpu_affinity_budget_is_explicit_and_does_not_modify_parent(monkeypatch):
    selected = []
    monkeypatch.setattr(
        subject.os,
        "sched_getaffinity",
        lambda _pid: {4, 8, 9} if not selected else selected[-1],
    )
    monkeypatch.setattr(
        subject.os, "sched_setaffinity", lambda _pid, mask: selected.append(mask)
    )
    assert subject._cpu_affinity(2) == {"applied": True, "logical_cpus": 2}
    assert selected == [{4, 8}]


def test_parent_import_does_not_import_native_or_control_modules():
    root = Path(subject.__file__).resolve().parents[1]
    script = f"import sys; sys.path.insert(0, {str(root)!r}); import tools.english_asr_benchmark; print(int(any(x in sys.modules for x in ('numpy', 'sherpa_onnx', 'faster_whisper', 'moonshine_voice', 'sounddevice', 'core', 'always_on_agent', 'torch'))))"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "0"
    assert completed.stderr == ""


def test_public_cli_errors_do_not_echo_private_arguments(tmp_path, capsys):
    assert subject.main(["--manifest", str(tmp_path / "private-manifest.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == '{"error":"english_asr_benchmark_unavailable","ok":false}\n'
    assert captured.err == ""
    with pytest.raises(SystemExit):
        subject.main(["--private-argument", "private reference words"])
    captured = capsys.readouterr()
    assert "private reference words" not in captured.err


def test_model_config_is_explicit_cpu_family_and_rejects_extra_backend_options(
    tmp_path,
):
    _manifest, model = _inputs(tmp_path)
    config = tmp_path / "models.json"
    entry = {
        "id": model.id,
        "artifacts": {key: str(path) for key, path in model.artifacts.items()},
    }
    config.write_text(json.dumps({"schema_version": 1, "models": [entry]}))
    models, digest = subject.load_models(config)
    assert models == (model,)
    assert digest == hashlib.sha256(config.read_bytes()).hexdigest()
    entry["provider"] = "cuda"
    config.write_text(json.dumps({"schema_version": 1, "models": [entry]}))
    with pytest.raises(subject.BenchmarkError):
        subject.load_models(config)
    entry.pop("provider")
    config.write_text(json.dumps({"schema_version": 1, "models": [entry, entry]}))
    with pytest.raises(subject.BenchmarkError):
        subject.load_models(config)


def test_moonshine_tuple_and_runtime_wheel_identity_are_measured(tmp_path):
    directory = tmp_path / "native-model"
    directory.mkdir()
    for name in subject.MOONSHINE_FILES:
        (directory / name).write_bytes(b"model artifact")
    wheel = tmp_path / "runtime.whl"
    wheel.write_bytes(b"runtime wheel")
    model = subject.Model(
        "moonshine-tiny-streaming-v015",
        {"directory": directory},
        Path(sys.executable),
        "tiny",
        wheel,
        hashlib.sha256(wheel.read_bytes()).hexdigest(),
    )
    assert subject.model_binding(model)["files"] == 8
    assert subject._runtime_binding(model)["wheel_sha256"] == model.runtime_wheel_sha256
    wheel.write_bytes(b"mutated wheel")
    with pytest.raises(subject.BenchmarkError):
        subject._runtime_binding(model)
    (directory / "extra-file").write_bytes(b"unexpected")
    with pytest.raises(subject.BenchmarkError):
        subject.model_binding(model)


def test_worker_failure_never_exposes_raw_native_streams_or_private_inputs(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    subject._write_new(
        request_path,
        {
            "manifest": str(manifest),
            "source_root": str(corpus.source_root),
            "model": {
                "id": model.id,
                "artifacts": {key: str(path) for key, path in model.artifacts.items()},
                "python": None,
                "arch": None,
                "runtime_wheel": None,
                "runtime_wheel_sha256": None,
                "ort_single_thread": False,
            },
            "corpus_digest": corpus.digest,
            "model_tuple": {"sha256": "0" * 64, "files": 2, "bytes": 1},
            "runtime_binding": {},
            "code_binding": {},
            "repeats": 1,
            "threads": 1,
            "timeout_sec": 5,
            "result": str(result_path),
        },
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(subject.__file__)),
            "--worker-request",
            str(request_path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert completed.stdout == completed.stderr == b""
    result = json.loads(result_path.read_bytes())
    assert result == {
        "model_id": model.id,
        "status": "worker_failed",
        "error_count": 1,
        "complete_corpus_coverage": False,
    }
    assert os.stat(result_path).st_mode & 0o777 == 0o600


def test_worker_network_hooks_refuse_python_connections(monkeypatch):
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(
            subject.socket.socket, name, getattr(subject.socket.socket, name)
        )
    monkeypatch.setattr(
        subject.socket, "create_connection", subject.socket.create_connection
    )
    subject._deny_network()
    with pytest.raises(subject.BenchmarkError):
        subject.socket.create_connection(("example.invalid", 443))
    with subject.socket.socket() as endpoint:
        with pytest.raises(subject.BenchmarkError):
            endpoint.connect(("example.invalid", 443))


def test_moonshine_thread_policy_does_not_claim_native_pool_budget(monkeypatch):
    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", "0")
    model = subject.Model("moonshine-tiny-streaming-v015", {})
    assert subject._thread_policy(model, 2) == {
        "requested_affinity_threads": 2,
        "native_thread_budget_supported": False,
        "configured_native_threads": None,
        "moonshine_single_thread_env": "0",
        "ort_single_thread": False,
    }
    assert (
        subject._thread_policy(subject.Model("sensevoice-en", {}), 2)[
            "configured_native_threads"
        ]
        == 2
    )
    count = subject._process_threads()
    assert count is None or count >= 1


def test_complete_payload_from_failed_process_is_not_accepted(tmp_path, monkeypatch):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    payload = subject.benchmark_cell(
        model,
        corpus,
        1,
        2,
        decoder_factory=lambda _m, _t: lambda _samples: "private reference words",
    )
    payload["cpu_affinity"] = {"applied": False, "logical_cpus": None}

    class FailedProcess:
        pid = 12345
        returncode = 1

        def wait(self, timeout):
            return 1

        def poll(self):
            return 1

    def popen(argv, **_kwargs):
        request = json.loads(Path(argv[-1]).read_text())
        subject._write_new(Path(request["result"]), payload)
        return FailedProcess()

    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    result = subject.run_cell(model, corpus, scratch, 1, 2, 5)
    assert result["status"] == "worker_failed"
    assert result["complete_corpus_coverage"] is False


def _complete_report(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    result = subject.benchmark_cell(
        model,
        corpus,
        1,
        2,
        decoder_factory=lambda _m, _t: lambda _samples: "private reference words",
    )
    result["cpu_affinity"] = {"applied": False, "logical_cpus": None}
    return result, corpus, model


@pytest.mark.parametrize(
    "path",
    [
        ("accuracy", "empty"),
        ("by_role", "command", "word_errors"),
        ("model_load_ms",),
        ("model_tuple", "files"),
        ("thread_policy", "configured_native_threads"),
        ("process_threads", "after_model_load"),
        ("model_input", "tail_padding_samples_per_clip"),
    ],
)
def test_scalar_fields_cannot_embed_private_dictionary_keys(tmp_path, path):
    result, corpus, model = _complete_report(tmp_path)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = {"PRIVATE_SENTINEL": 1}
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(
            result, model.id, 3, 1, source_seconds=corpus.seconds
        )


def test_typed_accuracy_bounds_and_ratios_are_independently_verified(tmp_path):
    result, corpus, model = _complete_report(tmp_path)
    subject._validate_cell_result(
        result,
        model.id,
        3,
        1,
        source_seconds=corpus.seconds,
        role_counts={"round_trip": 1, "barge": 1, "command": 1},
    )
    for key, bad in (
        ("empty", True),
        ("empty", 4),
        ("reference_words", 1000),
        ("wer", 0.25),
    ):
        forged = copy.deepcopy(result)
        forged["accuracy"][key] = bad
        with pytest.raises(subject.BenchmarkError):
            subject._validate_cell_result(
                forged, model.id, 3, 1, source_seconds=corpus.seconds
            )
    forged = copy.deepcopy(result)
    forged["repeat_disagreements"] = 1  # repeat1 has no repeat-comparison calls
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(
            forged, model.id, 3, 1, source_seconds=corpus.seconds
        )


def test_per_model_padding_receipt_keeps_rtf_denominator_original_audio(tmp_path):
    result, corpus, model = _complete_report(tmp_path)
    assert result["model_input"]["tail_padding_samples_per_clip"] == 0
    assert result["model_input"]["total_model_input_seconds"] == pytest.approx(
        corpus.seconds
    )
    for identifier in subject.ZIPFORMER_IDS:
        receipt = subject._input_receipt(identifier, corpus.seconds, 3, 3)
        assert receipt["tail_padding_samples_per_clip"] == 10560
        assert receipt["original_audio_seconds"] == pytest.approx(0.3)
        assert receipt["source_audio_seconds_across_calls"] == pytest.approx(0.9)
        assert receipt["total_model_input_seconds"] == pytest.approx(6.84)
        assert receipt["input_finished"] is True
        assert receipt["tail_padding_policy"] == "sherpa_streaming_file_flush"
    forged = copy.deepcopy(result)
    forged["model_input"] = subject._input_receipt("zipformer-en", corpus.seconds, 3, 1)
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(
            forged, model.id, 3, 1, source_seconds=corpus.seconds
        )
    forged = copy.deepcopy(result)
    forged["model_input"]["original_audio_seconds"] += 1
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(
            forged, model.id, 3, 1, source_seconds=corpus.seconds
        )


def test_process_thread_receipt_only_claims_two_load_samples(tmp_path, monkeypatch):
    samples = iter((2, 7))
    monkeypatch.setattr(subject, "_process_threads", lambda: next(samples))
    result, corpus, model = _complete_report(tmp_path)
    assert result["process_threads"] == {
        "before_model_load": 2,
        "after_model_load": 7,
        "maximum_sampled": 7,
    }
    subject._validate_cell_result(result, model.id, 3, 1, source_seconds=corpus.seconds)
    result["process_threads"]["maximum_sampled"] = 8
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(
            result, model.id, 3, 1, source_seconds=corpus.seconds
        )


def _moon_model(tmp_path, *, selected=False):
    directory = tmp_path / "moon-model"
    directory.mkdir()
    for name in subject.MOONSHINE_FILES:
        (directory / name).write_bytes(b"model artifact")
    return subject.Model(
        "moonshine-tiny-streaming-v015",
        {"directory": directory},
        Path(sys.executable),
        "tiny",
        ort_single_thread=selected,
    )


def _affinity_sample(*, outside=0, unknown=0, failures=0):
    return {
        "successful": True,
        "threads_observed": 2,
        "unknown_threads": unknown,
        "read_failures": failures,
        "outside_caller": outside,
        "union_logical_cpus": 3 if outside else 2,
    }


def test_moonshine_single_thread_flag_is_strict_explicit_and_digest_bound(tmp_path):
    model = _moon_model(tmp_path)
    config = tmp_path / "models.json"
    entry = {
        "id": model.id,
        "kind": "moonshine-native",
        "model_dir": str(model.artifacts["directory"]),
        "arch": "tiny",
        "python": sys.executable,
    }
    config.write_text(json.dumps({"schema_version": 1, "models": [entry]}))
    loaded, stock_digest = subject.load_models(config)
    assert loaded[0].ort_single_thread is False
    entry["ort_single_thread"] = True
    config.write_text(json.dumps({"schema_version": 1, "models": [entry]}))
    loaded, selected_digest = subject.load_models(config)
    assert loaded[0].ort_single_thread is True
    assert selected_digest != stock_digest
    for invalid in (1, 0, "true", None):
        entry["ort_single_thread"] = invalid
        config.write_text(json.dumps({"schema_version": 1, "models": [entry]}))
        with pytest.raises(subject.BenchmarkError):
            subject.load_models(config)


def test_selected_thread_policy_reports_fixed_one_not_arbitrary_native_budget(
    tmp_path, monkeypatch
):
    model = _moon_model(tmp_path, selected=True)
    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", "1")
    policy = subject._thread_policy(model, 2)
    assert policy["configured_native_threads"] == 1
    assert policy["native_thread_budget_supported"] is False
    assert policy["requested_affinity_threads"] == 2
    assert policy["ort_single_thread"] is True
    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", "0")
    with pytest.raises(subject.BenchmarkError):
        subject._thread_policy(model, 2)


@pytest.mark.parametrize("selected,env", [(False, "0"), (True, "1")])
def test_child_env_and_request_use_only_selected_flag(
    tmp_path, monkeypatch, selected, env
):
    manifest, _base = _inputs(tmp_path)
    model = _moon_model(tmp_path, selected=selected)
    corpus = subject.load_corpus(manifest)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    observed = []

    class Process:
        pid = 12345

        def __init__(self):
            self.running = True

        def poll(self):
            return None if self.running else 0

        def wait(self, timeout):
            if timeout == 5:
                raise subprocess.TimeoutExpired("synthetic", 5)
            self.running = False
            return 0

    def popen(argv, **kwargs):
        observed.append((json.loads(Path(argv[-1]).read_text()), kwargs["env"]))
        return Process()

    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", "1" if not selected else "0")
    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    monkeypatch.setattr(subject.os, "killpg", lambda *_args: None)
    result = subject.run_cell(model, corpus, scratch, 1, 2, 5)
    assert result["status"] == "worker_timeout"
    request, child_env = observed[0]
    assert request["model"]["ort_single_thread"] is selected
    assert child_env["MOONSHINE_ORT_SINGLE_THREAD"] == env


def test_native_masks_are_sampled_without_exposing_tids_or_masks(monkeypatch):
    import io

    entries = [
        SimpleNamespace(name="101", path="/synthetic/101"),
        SimpleNamespace(name="102", path="/synthetic/102"),
    ]

    class Directory:
        def __enter__(self):
            return iter(entries)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(subject.os, "scandir", lambda _path: Directory())
    statuses = {
        "/synthetic/101/status": "Threads: 2\nCpus_allowed_list:\t2-3\n",
        "/synthetic/102/status": "Threads: 2\nCpus_allowed_list:\t4,6-7\n",
    }
    monkeypatch.setattr(
        subject,
        "open",
        lambda path, *_a, **_kw: io.StringIO(statuses[str(path)]),
        raising=False,
    )
    sample = subject._sample_thread_affinity({2, 3})
    assert sample == {
        "successful": True,
        "threads_observed": 2,
        "unknown_threads": 0,
        "read_failures": 0,
        "outside_caller": 1,
        "union_logical_cpus": 5,
    }
    assert "synthetic" not in json.dumps(sample)
    collector = subject._AffinitySamples({2, 3}, lambda _mask: sample)
    with pytest.raises(subject.NativeAffinityViolation):
        collector.observe()
    receipt = collector.receipt()
    assert receipt["outside_caller_mask_observations"] == 1
    assert receipt["maximum_observed_logical_cpus"] == 5
    assert receipt["sampled_compliant"] is False
    subject._validate_thread_affinity(receipt, 4, complete=False)


def test_unknown_native_masks_are_recorded_without_claiming_sampled_compliance(
    monkeypatch,
):
    import io

    entries = [
        SimpleNamespace(name="101", path="/synthetic/101"),
        SimpleNamespace(name="102", path="/synthetic/102"),
    ]

    class Directory:
        def __enter__(self):
            return iter(entries)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(subject.os, "scandir", lambda _path: Directory())

    def read(path, *_args, **_kwargs):
        if str(path).endswith("102/status"):
            raise OSError()
        return io.StringIO("Threads: 2\n")  # absent affinity field

    monkeypatch.setattr(subject, "open", read, raising=False)
    sample = subject._sample_thread_affinity({2, 3})
    assert sample["unknown_threads"] == 2
    assert sample["read_failures"] == 1
    collector = subject._AffinitySamples({2, 3}, lambda _mask: sample)
    collector.observe()
    assert collector.receipt()["sampled_compliant"] is False


def test_sampled_violation_refuses_cell_before_decode_and_marks_before_close(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    order = []

    class Decoder:
        def __call__(self, _samples):
            order.append("decode")
            return "private reference words"

        def close(self):
            order.append("close")

    with pytest.raises(subject.NativeAffinityViolation):
        subject.benchmark_cell(
            model,
            corpus,
            1,
            2,
            decoder_factory=lambda *_args: Decoder(),
            caller_mask={2, 3},
            affinity_sampler=lambda _mask: _affinity_sample(outside=1),
            affinity_failure_sink=lambda _receipt: order.append("marker"),
        )
    assert order == ["marker", "close"]


def test_all_native_sampling_covers_load_and_each_decode_with_typed_receipt(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    calls = []

    def sampler(mask):
        assert mask == {2, 3}
        calls.append(True)
        return _affinity_sample()

    result = subject.benchmark_cell(
        model,
        corpus,
        2,
        2,
        decoder_factory=lambda *_args: lambda _pcm: "private reference words",
        caller_mask={2, 3},
        affinity_sampler=sampler,
    )
    result["cpu_affinity"] = {"applied": True, "logical_cpus": 2}
    assert len(calls) == 7
    receipt = result["thread_affinity"]
    assert receipt["sample_attempts"] == receipt["successful_samples"] == 7
    assert receipt["sampled_compliant"] is True
    assert receipt["continuous_observation"] is receipt["cgroup_isolation"] is False
    subject._validate_cell_result(result, model.id, 3, 2)
    forged = copy.deepcopy(result)
    forged["thread_affinity"]["unknown_thread_observations"] = {"PRIVATE_SENTINEL": 1}
    with pytest.raises(subject.BenchmarkError):
        subject._validate_cell_result(forged, model.id, 3, 2)


def test_affinity_violation_after_decode_stops_remaining_calls(tmp_path):
    manifest, model = _inputs(tmp_path)
    corpus = subject.load_corpus(manifest)
    calls = []
    samples = iter((_affinity_sample(), _affinity_sample(outside=1)))

    def decode(_pcm):
        calls.append(True)
        return "private reference words"

    with pytest.raises(subject.NativeAffinityViolation):
        subject.benchmark_cell(
            model,
            corpus,
            3,
            2,
            decoder_factory=lambda *_args: decode,
            caller_mask={2, 3},
            affinity_sampler=lambda _mask: next(samples),
        )
    assert calls == [True]
