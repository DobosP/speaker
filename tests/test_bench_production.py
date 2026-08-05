from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from types import SimpleNamespace

import pytest

from core.metrics import (
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    SPEECH_END,
    TTS_FIRST_AUDIO,
    MetricsRecorder,
    TurnRecord,
)
from tools.bench import report, runner
from tools.bench.production import (
    BoundOllamaIdentity,
    PreparedProduction,
    ProductionInputError,
    active_artifact_identity,
    bind_ollama_identity,
    load_production_corpus,
    prepare_production,
    verify_ollama_identity,
    verify_production_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_schema_v2_corpus(root: Path, *, text: str = "hello world") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = b"".join(struct.pack("<f", value) for value in (0.1, -0.1, 0.2, -0.2))
    audio = root / "case.f32le"
    audio.write_bytes(raw)
    corpus = root / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "purpose": "deterministic production bench test",
                "provenance": {
                    "kind": "public-voice-v1",
                    "suite": "bench-test",
                    "manifest_sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                    "source_set_sha256": "c" * 64,
                },
                "cases": [
                    {
                        "id": "case-1",
                        "file": audio.name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "samples": 4,
                        "expected_text": text,
                        "assertion": "transcript",
                        "commands": [],
                        "tags": ["clean"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return corpus


def _identity_snapshot(role_models: dict[str, str], _config: dict, *, suffix="1"):
    models = {
        model: {
            "model": model,
            "verification": "ollama_blob_effective_config",
            "required": True,
            "ok": True,
            "blob_sha256": suffix * 64,
            "effective_config_sha256": ("2" if suffix == "1" else "3") * 64,
        }
        for model in set(role_models.values())
    }
    return {"role_models": role_models, "models": models, "ok": True}


def _prepared_readers():
    return {
        "artifact_reader": lambda _config, **_kwargs: ("d" * 64, 5, 7, 1024),
        "repository_reader": lambda _root: {
            "revision": "e" * 40,
            "dirty": False,
        },
        "runtime_reader": lambda _config: {
            "python": {"version": "3.12.0", "executable_sha256": "f" * 64},
            "distributions": {},
        },
    }


def _patch_production_cli_dependencies(monkeypatch, production_run):
    import core.sysinfo as sysinfo
    import tools.bench.production as production

    prepared = PreparedProduction(
        config={"llm": {"backend": "ollama"}},
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=1,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=1,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=8,
        active_artifact_bytes=1024,
        source_revision="6" * 40,
        runtime_identities={"python": {"executable_sha256": "7" * 64}},
    )
    corpus = SimpleNamespace(cases=())
    identity = BoundOllamaIdentity(
        sha256="8" * 64,
        role_contracts={
            "main": {
                "blob_sha256": "9" * 64,
                "effective_config_sha256": "a" * 64,
            },
            "fast": {
                "blob_sha256": "b" * 64,
                "effective_config_sha256": "c" * 64,
            },
        },
        snapshot={},
    )

    class Monitor:
        def __init__(self, *, interval):
            self.interval = interval

        def start(self):
            return None

        def mark(self, _name):
            return {}

        def stop(self):
            return {"samples": 0, "peak": {}}

    monkeypatch.setattr(sysinfo, "SystemMonitor", Monitor)
    monkeypatch.setattr(production, "load_production_corpus", lambda _path: corpus)
    monkeypatch.setattr(production, "prepare_production", lambda **_kwargs: prepared)
    monkeypatch.setattr(production, "bind_ollama_identity", lambda _config: identity)
    monkeypatch.setattr(production, "verify_ollama_identity", lambda *_args: None)
    monkeypatch.setattr(production, "verify_production_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        production,
        "corpus_identity",
        lambda _corpus: {"schema_version": 2, "sha256": "d" * 64, "case_count": 1},
    )
    monkeypatch.setattr(runner, "run_production", production_run)


def test_production_corpus_requires_receipt_aware_schema(tmp_path):
    path = _write_schema_v2_corpus(tmp_path)

    loaded = load_production_corpus(path)

    assert loaded.schema_version == 2
    assert loaded.cases[0].expected_text == "hello world"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("provenance")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProductionInputError):
        load_production_corpus(path)


def test_prepare_binds_atomic_profile_safety_and_close_time_config(tmp_path):
    config_path = tmp_path / "config.json"
    configured = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    profile_llm = configured["device_profiles"]["desktop_gpu_4090"]["llm"]
    profile_llm["host"] = "https://remote.invalid"
    profile_llm["router_model"] = "unbound-router:test"
    profile_llm["cloud"] = {"enabled": True, "strategy": "hedge"}
    config_path.write_text(json.dumps(configured), encoding="utf-8")
    local_path = tmp_path / "missing-local.json"
    readers = _prepared_readers()
    hashed_configs = []

    def artifact_reader(config, **_kwargs):
        hashed_configs.append(config)
        return "d" * 64, 5, 7, 1024

    readers["artifact_reader"] = artifact_reader

    prepared = prepare_production(
        config_path=config_path,
        local_config_path=local_path,
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        **readers,
    )

    assert prepared.config["memory"]["backend"] == "session"
    assert prepared.config["llm"]["host"] == "http://127.0.0.1:11434"
    assert prepared.config["llm"]["cloud"]["enabled"] is False
    assert prepared.config["llm"]["cloud"]["strategy"] == "local_only"
    assert prepared.config["llm"]["router_model"] is None
    assert prepared.config["llm"]["live_routing"] is False
    assert prepared.replay_safety_schema_version == 2
    assert prepared.config["sherpa"]["asr_final_model"] == str(
        tmp_path / "pretrained_models/sherpa/sense_voice/model.int8.onnx"
    )
    assert prepared.config["sherpa"]["asr_final_tokens"] == str(
        tmp_path / "pretrained_models/sherpa/sense_voice/tokens.txt"
    )
    assert hashed_configs == [prepared.config]
    assert all(
        prepared.config[name]["enabled"] is False
        for name in ("obsidian", "reminders", "trusted_apps", "web_search")
    )
    assert prepared.final_stt_profile == "sense-voice"
    assert prepared.active_artifacts_sha256 == "d" * 64
    assert prepared.source_revision == "e" * 40
    corpus = load_production_corpus(_write_schema_v2_corpus(tmp_path / "corpus"))
    verify_production_inputs(
        prepared,
        corpus,
        config_path=config_path,
        local_config_path=local_path,
        **readers,
    )

    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["device_profiles"]["desktop_gpu_4090"]["llm"]["options"][
        "num_ctx"
    ] = 1234
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ProductionInputError):
        verify_production_inputs(
            prepared,
            corpus,
            config_path=config_path,
            local_config_path=local_path,
            **readers,
        )


def test_prepare_normalizes_singleton_and_list_artifact_paths(tmp_path):
    config_path = tmp_path / "config.json"
    configured = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    configured["sherpa"]["punct_model"] = " models/punct.onnx "
    configured["sherpa"]["tts_lexicon"] = " lexicon/a.txt, /models/b.txt "
    config_path.write_text(json.dumps(configured), encoding="utf-8")
    seen = []
    readers = _prepared_readers()

    def artifact_reader(config, **_kwargs):
        seen.append(config)
        return "d" * 64, 5, 7, 1024

    readers["artifact_reader"] = artifact_reader
    prepared = prepare_production(
        config_path=config_path,
        local_config_path=tmp_path / "missing-local.json",
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        **readers,
    )

    assert prepared.config["sherpa"]["punct_model"] == str(
        tmp_path / "models/punct.onnx"
    )
    assert prepared.config["sherpa"]["tts_lexicon"] == (
        f"{tmp_path / 'lexicon/a.txt'},/models/b.txt"
    )
    assert seen == [prepared.config]


@pytest.mark.parametrize(
    ("field_name", "raw_value"),
    (("punct_model", "   "), ("tts_lexicon", ", ,")),
)
def test_prepare_rejects_truthy_artifact_values_with_no_path(
    tmp_path, field_name, raw_value
):
    config_path = tmp_path / "config.json"
    configured = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    configured["sherpa"][field_name] = raw_value
    config_path.write_text(json.dumps(configured), encoding="utf-8")

    with pytest.raises(ProductionInputError):
        prepare_production(
            config_path=config_path,
            local_config_path=tmp_path / "missing-local.json",
            device_profile="desktop_gpu_4090",
            final_stt_profile="sense-voice",
            **_prepared_readers(),
        )


def test_active_audio_artifact_digest_binds_file_and_tree_bytes(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model-v1")
    data = tmp_path / "data"
    data.mkdir()
    (data / "voice.bin").write_bytes(b"voice-v1")
    config = {
        "sherpa": {
            "asr_encoder": str(model),
            "asr_tokens": str(model),
            "vad_model": str(model),
            "tts_model": str(model),
            "tts_tokens": str(model),
            "tts_data_dir": str(data),
            "asr_final_backend": "sense_voice",
            "asr_final_model": str(model),
            "asr_final_tokens": str(model),
        }
    }

    first = active_artifact_identity(config, config_root=tmp_path)
    (data / "voice.bin").write_bytes(b"voice-v2")
    second = active_artifact_identity(config, config_root=tmp_path)

    assert first[0] != second[0]
    assert first[1:] == second[1:]

    (data / "unsafe-link").symlink_to(model)
    with pytest.raises(ProductionInputError):
        active_artifact_identity(config, config_root=tmp_path)


def test_active_audio_artifact_tree_scan_is_entry_bounded(monkeypatch, tmp_path):
    import tools.bench.production as production

    data = tmp_path / "data"
    data.mkdir()
    (data / "one.bin").write_bytes(b"one")
    (data / "two.bin").write_bytes(b"two")
    monkeypatch.setattr(production, "_MAX_ARTIFACT_ENTRIES", 1)

    with pytest.raises(ProductionInputError):
        active_artifact_identity(
            {"sherpa": {"tts_data_dir": str(data)}},
            config_root=tmp_path,
        )


def test_ollama_main_fast_contract_is_bound_and_reverified():
    config = {
        "llm": {
            "backend": "ollama",
            "main_model": "main:test",
            "fast_model": "fast:test",
        }
    }
    before = bind_ollama_identity(config, snapshot_reader=_identity_snapshot)

    assert set(before.role_contracts) == {"main", "fast"}
    verify_ollama_identity(before, config, snapshot_reader=_identity_snapshot)
    with pytest.raises(ProductionInputError):
        verify_ollama_identity(
            before,
            config,
            snapshot_reader=lambda roles, cfg: _identity_snapshot(
                roles, cfg, suffix="4"
            ),
        )


def test_run_production_uses_shared_factories_and_keeps_cases_aligned(monkeypatch):
    import core.app as core_app
    import core.engines.file_replay as file_replay
    import core.llm_factory as llm_factory
    import core.routing as routing

    calls: dict[str, object] = {}

    class Engine:
        def __init__(self, config):
            self.config = config
            self.finals: list[str] = []
            self.spoken: list[str] = []
            self.last_final = ""
            self.runtime = None

        def replay_samples(self, samples, sample_rate):
            assert samples.shape == (4,)
            assert sample_rate == 16_000
            metrics = self.runtime.metrics
            metrics.mark(SPEECH_END)
            metrics.mark(ASR_FINAL)
            metrics.mark(LLM_FIRST_TOKEN)
            metrics.mark(TTS_FIRST_AUDIO)
            self.finals.append("hello world")
            self.spoken.append("private assistant answer")

    class Runtime:
        def __init__(self, engine):
            self.engine = engine
            self.metrics = MetricsRecorder()
            self.stopped = False
            self.warm_ready = SimpleNamespace(wait=lambda *, timeout: timeout > 0)
            engine.runtime = self

        def start(self, *, run_bus):
            calls["run_bus"] = run_bus

        def wait_idle(self, *, timeout):
            calls["timeout"] = timeout
            return True

        def stop(self):
            self.stopped = True

    def fake_build_llms(args, config):
        from core.llm import OllamaLLM

        calls["llm_args"] = args
        calls["llm_config"] = config
        headers = dict(args.ollama_client_headers)
        return (
            OllamaLLM("main:test", client=object(), client_headers=headers),
            OllamaLLM("fast:test", client=object(), client_headers=headers),
        )

    def fake_build_runtime(config, **kwargs):
        calls["runtime_config"] = config
        calls["runtime_kwargs"] = kwargs
        return Runtime(kwargs["engine"])

    monkeypatch.setattr(file_replay, "FileReplayEngine", Engine)
    monkeypatch.setattr(llm_factory, "build_llms", fake_build_llms)
    monkeypatch.setattr(routing, "build_router", lambda config: ("router", config))
    monkeypatch.setattr(core_app, "build_runtime", fake_build_runtime)
    case = SimpleNamespace(
        case_id="case-1",
        audio_bytes=b"".join(struct.pack("<f", value) for value in (0.1, 0.2, 0.3, 0.4)),
        expected_text="hello world",
        assertion="transcript",
        commands=(),
        forbidden_commands=(),
    )
    corpus = SimpleNamespace(cases=(case,))
    config = {
        "llm": {"backend": "ollama"},
        "sherpa": {
            "asr_encoder": "encoder",
            "asr_tokens": "tokens",
            "vad_model": "vad",
            "tts_model": "tts",
            "tts_tokens": "tts-tokens",
        },
    }
    built: list[bool] = []

    result = runner.run_production(
        corpus,
        config,
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=1,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=1,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=5,
        active_artifact_bytes=100,
        source_revision="6" * 40,
        runtime_identities={},
        ollama_identity_sha256="7" * 64,
        ollama_role_contracts={},
        after_build=lambda: built.append(True),
    )

    assert built == [True]
    assert calls["run_bus"] is False
    assert calls["runtime_kwargs"]["llm"].model == "main:test"
    assert calls["runtime_kwargs"]["fast_llm"].model == "fast:test"
    assert result.samples[0].transcript == "hello world"
    assert result.samples[0].reference == "hello world"
    assert result.samples[0].record.final_to_first_token is not None


def test_production_accuracy_counts_each_wer_operation_once():
    def sample(reference: str, transcript: str) -> runner.TurnSample:
        return runner.TurnSample(
            name="private-case",
            expectation="transcript",
            transcript=transcript,
            responded=False,
            record=TurnRecord(),
            reference=reference,
            assertion="transcript",
        )

    accuracy = report.production_accuracy(
        (
            sample("one", "two"),
            sample("one", "one extra"),
            sample("one two", "one"),
        )
    )
    transcript = accuracy["transcript"]

    assert transcript == {
        "case_count": 3,
        "exact_match_count": 0,
        "word_error_rate": 0.75,
        "word_errors": 3,
        "substitutions": 1,
        "insertions": 1,
        "deletions": 1,
        "reference_words": 4,
        "hypothesis_words": 4,
    }


def test_production_report_is_private_exclusive_and_replay_truthful(tmp_path):
    secret = "/private/paul secret transcript"
    sample = runner.TurnSample(
        name=secret,
        expectation="transcript",
        transcript="hello there",
        responded=True,
        record=TurnRecord(
            stamps={ASR_FINAL: 1.0, LLM_FIRST_TOKEN: 1.2, TTS_FIRST_AUDIO: 1.5}
        ),
        reference="hello world",
        assertion="transcript",
        replay_decode_seconds=0.4,
        replay_to_idle_seconds=0.8,
    )
    run = runner.ProductionRun(
        samples=(sample,),
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=1,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=1,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=8,
        active_artifact_bytes=1024,
        source_revision="6" * 40,
        runtime_identities={"python": {"executable_sha256": "7" * 64}},
        ollama_identity_sha256="8" * 64,
        ollama_role_contracts={
            "main": {"blob_sha256": "9" * 64, "effective_config_sha256": "a" * 64},
            "fast": {"blob_sha256": "b" * 64, "effective_config_sha256": "c" * 64},
        },
    )
    destination = tmp_path / "private-report"

    index, payload = report.write_production_reports(
        destination,
        run,
        {"schema_version": 2, "sha256": "d" * 64, "case_count": 1},
        {"samples": 2, "peak": {"proc_rss_mb": 123.0, "gpu_mem_used_mb": 456.0}},
        sample_interval_sec=1.0,
    )

    serialized = (destination / "summary.json").read_text(encoding="utf-8")
    assert secret not in serialized
    assert "hello there" not in serialized
    assert "hello world" not in serialized
    assert set(payload["latency"]) == {
        "complete-PCM→replay-decode-return",
        "complete-PCM→runtime-idle",
        "selected-final→token",
        "token→TTS-clip-ready",
    }
    assert payload["latency"]["selected-final→token"]["p50_seconds"] == 0.2
    assert any("audible" in item for item in payload["limitations"])
    assert payload["method"]["case_isolation"] == (
        "fresh-engine-router-runtime-session-per-case"
    )
    assert index.exists()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(index.stat().st_mode) == 0o600
    assert stat.S_IMODE((destination / "summary.json").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        report.write_production_reports(
            destination,
            run,
            {"schema_version": 2, "sha256": "d" * 64, "case_count": 1},
            {},
            sample_interval_sec=1.0,
        )

    linked = tmp_path / "linked-report"
    linked.symlink_to(destination, target_is_directory=True)
    with pytest.raises((FileExistsError, OSError)):
        report.write_production_reports(
            linked,
            run,
            {"schema_version": 2, "sha256": "d" * 64, "case_count": 1},
            {},
            sample_interval_sec=1.0,
        )


def test_production_cli_never_publishes_private_case_text_or_input_path(
    monkeypatch, tmp_path, capsys
):
    from tools.bench.__main__ import main

    canaries = (
        "PRIVATE_CASE_ID_CANARY",
        "PRIVATE_REFERENCE_CANARY",
        "PRIVATE_HYPOTHESIS_CANARY",
        "PRIVATE_INPUT_PATH_CANARY",
    )
    sample = runner.TurnSample(
        name=canaries[0],
        expectation="transcript",
        transcript=canaries[2],
        responded=False,
        record=TurnRecord(),
        reference=canaries[1],
        assertion="transcript",
        replay_decode_seconds=0.1,
        replay_to_idle_seconds=0.2,
    )
    production_run = runner.ProductionRun(
        samples=(sample,),
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=1,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=1,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=8,
        active_artifact_bytes=1024,
        source_revision="6" * 40,
        runtime_identities={"python": {"executable_sha256": "7" * 64}},
        ollama_identity_sha256="8" * 64,
        ollama_role_contracts={},
    )
    _patch_production_cli_dependencies(
        monkeypatch,
        lambda *_args, **_kwargs: production_run,
    )
    output_root = tmp_path / "reports"

    assert main(
        [
            "--mode",
            "production",
            "--device",
            "desktop_gpu_4090",
            "--corpus",
            f"/private/{canaries[3]}/corpus.json",
            "--final-stt-profile",
            "sense-voice",
            "--out",
            str(output_root),
            "--md-summary",
            "",
        ]
    ) == 0

    captured = capsys.readouterr()
    destination = output_root / "desktop_gpu_4090"
    published = captured.out + captured.err
    published += (destination / "summary.json").read_text(encoding="utf-8")
    published += (destination / "index.html").read_text(encoding="utf-8")
    for canary in canaries:
        assert canary not in published


def test_production_cli_reports_runtime_failure_without_private_detail(
    monkeypatch, tmp_path, capsys
):
    from tools.bench.__main__ import main

    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("PRIVATE_RUNTIME_DETAIL_CANARY")

    _patch_production_cli_dependencies(monkeypatch, fail_replay)
    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--mode",
                "production",
                "--device",
                "desktop_gpu_4090",
                "--corpus",
                "/private/PRIVATE_INPUT_PATH_CANARY/corpus.json",
                "--final-stt-profile",
                "sense-voice",
                "--out",
                str(tmp_path / "reports"),
                "--md-summary",
                "",
            ]
        )

    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert "production replay failed" in captured.err
    assert "PRIVATE_RUNTIME_DETAIL_CANARY" not in captured.out + captured.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--fake", "--corpus", "PRIVATE_INPUT_PATH_CANARY/corpus.json"],
        ["--fake", "--final-stt-profile", "sense-voice"],
        ["--fake", "--device", "desktop_gpu_4090"],
        ["--fake", "--config", "PRIVATE_INPUT_PATH_CANARY/config.json"],
    ],
)
def test_legacy_fake_rejects_production_only_arguments(arguments, capsys):
    from tools.bench.__main__ import main

    with pytest.raises(SystemExit) as failure:
        main(arguments)

    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert "PRIVATE_INPUT_PATH_CANARY" not in captured.out + captured.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--mode", "production"],
        ["--mode", "production", "--device", "desktop_gpu_4090"],
        [
            "--mode",
            "production",
            "--device",
            "desktop_gpu_4090",
            "--corpus",
            "corpus.json",
        ],
    ],
)
def test_production_cli_requires_explicit_device_corpus_and_profile(arguments):
    from tools.bench.__main__ import main

    with pytest.raises(SystemExit) as failure:
        main(arguments)
    assert failure.value.code == 2
