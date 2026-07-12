from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from core.minicpm_identity import (
    MINICPM_Q8_CONTRACT,
    validate_minicpm_modelfile,
    verify_minicpm_q8_identity,
)
import tools.setup_minicpm as setup
from tools.setup_minicpm import (
    DEFAULT_MODELFILE,
    LOCAL_MODEL,
    SOURCE_MODEL,
    MiniCPMIdentityError,
    provision,
)


_REPO = Path(__file__).resolve().parents[1]


def _show_pair(
    *,
    alias_digest: str = MINICPM_Q8_CONTRACT.blob_sha256,
    source_digest: str = MINICPM_Q8_CONTRACT.blob_sha256,
    quantization: str = "Q8_0",
    template: str = MINICPM_Q8_CONTRACT.template,
    parameters: str | None = None,
    alias_modelfile: str | None = None,
    source_modelfile: str | None = None,
):
    alias = {
        "modelfile": alias_modelfile or f"FROM /models/sha256-{alias_digest}",
        "template": template,
        "parameters": parameters or (
            'stop "<|im_end|>"\n'
            'stop "</s>"\n'
            "temperature 0.7\n"
            "top_p 0.95\n"
            "num_ctx 8192"
        ),
        "details": {"quantization_level": quantization},
    }
    source = {
        "modelfile": source_modelfile or f"FROM /models/sha256-{source_digest}"
    }
    return lambda model: alias if model == LOCAL_MODEL else source


def test_provision_pulls_creates_then_verifies_canonical_identity():
    calls = []

    identity = provision(
        runner=lambda command, *, check: calls.append((command, check)),
        show=_show_pair(),
    )

    assert calls == [
        (["ollama", "pull", SOURCE_MODEL], True),
        (["ollama", "create", LOCAL_MODEL, "-f", str(DEFAULT_MODELFILE)], True),
    ]
    assert identity.ok and identity.alias_blob_sha256 == MINICPM_Q8_CONTRACT.blob_sha256


def test_provision_can_rebuild_alias_without_pull_then_verifies():
    calls = []

    provision(
        pull=False,
        runner=lambda command, *, check: calls.append((command, check)),
        show=_show_pair(),
    )

    assert calls == [
        (["ollama", "create", LOCAL_MODEL, "-f", str(DEFAULT_MODELFILE)], True)
    ]


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("source_model", "hf.co/openbmb/MiniCPM5-1B-GGUF:Q4_K_M"),
        ("local_model", "minicpm5-1b:latest"),
        ("modelfile", Path("/tmp/wrong-minicpm-Modelfile")),
    ),
)
def test_provision_rejects_noncanonical_identity_before_commands(override, value):
    kwargs = {override: value}
    with pytest.raises(MiniCPMIdentityError):
        provision(
            **kwargs,
            runner=lambda *_args, **_kwargs: pytest.fail("runner was called"),
            show=lambda _model: pytest.fail("show was called"),
        )


def test_modelfile_validation_fails_closed_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_minicpm_modelfile(tmp_path / "missing")


@pytest.mark.parametrize(
    "show",
    (
        _show_pair(alias_digest="a" * 64),
        _show_pair(source_digest="b" * 64),
        _show_pair(quantization="Q4_K_M"),
        _show_pair(template="{{ .Prompt }}"),
        _show_pair(template=f"\n{MINICPM_Q8_CONTRACT.template}\n"),
        _show_pair(
            parameters=(
                'stop "<|im_end|>"\nstop "</s>"\nstop "<extra>"\n'
                "temperature 0.7\ntop_p 0.95\nnum_ctx 8192"
            )
        ),
        _show_pair(
            parameters=(
                'stop "<|im_end|>"\nstop "</s>"\n'
                "temperature 0.7\ntop_p 0.95\nnum_ctx 8192\nnum_predict 64"
            )
        ),
        _show_pair(
            alias_modelfile=(
                f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}\n"
                'SYSTEM "unexpected behavior override"'
            )
        ),
        _show_pair(
            alias_modelfile=(
                f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}\n"
                "FROM /models/sha256-" + "a" * 64
            )
        ),
        _show_pair(
            source_modelfile=(
                f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}\n"
                "FROM /models/sha256-" + "b" * 64
            )
        ),
        _show_pair(
            alias_modelfile=(
                f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}\n"
                'TEMPLATE ""'
            )
        ),
    ),
)
def test_post_create_identity_mismatch_fails_closed(show):
    with pytest.raises(MiniCPMIdentityError) as exc:
        provision(runner=lambda *_args, **_kwargs: None, show=show)
    assert exc.value.identity is not None
    assert exc.value.identity.ok is False


def test_post_create_show_error_fails_closed():
    def boom(_model):
        raise RuntimeError("synthetic show failure")

    with pytest.raises(MiniCPMIdentityError, match="synthetic show failure"):
        provision(runner=lambda *_args, **_kwargs: None, show=boom)


def test_nonbehavioral_ollama_license_metadata_is_allowed():
    show = _show_pair(
        alias_modelfile=(
            f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}\n"
            'LICENSE """Apache License 2.0\nCopyright OpenBMB\n"""'
        )
    )

    identity = verify_minicpm_q8_identity(show=show)

    assert identity.ok


def test_command_failure_skips_post_create_show():
    def fail_create(command, *, check):
        assert check is True
        if command[1] == "create":
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        provision(
            runner=fail_create,
            show=lambda _model: pytest.fail("show was called"),
        )


def test_cli_never_prints_ready_on_identity_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        setup,
        "provision",
        lambda **_kwargs: (_ for _ in ()).throw(
            MiniCPMIdentityError("synthetic identity mismatch")
        ),
    )
    with pytest.raises(SystemExit) as exc:
        setup.main([])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "ready" not in captured.out.lower()
    assert "identity mismatch" in captured.err.lower()


def test_identity_client_construction_error_is_controlled(monkeypatch, capsys):
    monkeypatch.setattr(
        setup,
        "_default_show",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic client failure")),
    )

    with pytest.raises(SystemExit) as exc:
        setup.main(["--verify-only"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "identity inspector unavailable" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_default_identity_client_is_explicitly_bound_to_ollama_host(monkeypatch):
    calls = []

    class Client:
        def show(self, model):
            return model

    def factory(**kwargs):
        calls.append(kwargs)
        return Client()

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    show = setup._default_show(client_factory=factory)

    assert calls == [{"timeout": 5.0, "host": "http://ollama:11434"}]
    assert show("model") == "model"


def test_verify_only_needs_no_deploy_file_or_provision(monkeypatch, capsys, tmp_path):
    identity = verify_minicpm_q8_identity(show=_show_pair())
    monkeypatch.setattr(
        setup,
        "DEFAULT_MODELFILE",
        tmp_path / "absent-deploy" / "Modelfile",
    )
    monkeypatch.setattr(setup, "verify_installed", lambda: identity)
    monkeypatch.setattr(
        setup,
        "validate_minicpm_modelfile",
        lambda _path: pytest.fail("Modelfile was read"),
    )
    monkeypatch.setattr(
        setup,
        "provision",
        lambda **_kwargs: pytest.fail("provision was called"),
    )

    assert setup.main(["--verify-only"]) == 0
    assert "identity verified" in capsys.readouterr().out.lower()


def test_shipped_ollama_profiles_use_supported_alias_and_keep_vision_main():
    cfg = json.loads((_REPO / "config.json").read_text(encoding="utf-8"))
    assert cfg["llm"]["fast_model"] == LOCAL_MODEL
    for profile in (
        "desktop",
        "desktop_gpu_4090",
        "macbook_m_series",
        "cpu_laptop",
        "open_speaker",
    ):
        llm = cfg["device_profiles"][profile]["llm"]
        assert llm["fast_model"] == LOCAL_MODEL
        assert llm["main_model"].startswith("gemma3:")


def test_committed_modelfile_matches_the_production_contract():
    validate_minicpm_modelfile()
    text = DEFAULT_MODELFILE.read_text(encoding="utf-8")
    assert f"FROM {SOURCE_MODEL}" in text
    assert MINICPM_Q8_CONTRACT.template in text


def test_modelfile_contract_rejects_additional_behavior_parameters(tmp_path):
    changed = tmp_path / "Modelfile"
    changed.write_text(
        DEFAULT_MODELFILE.read_text(encoding="utf-8")
        + "\nPARAMETER num_predict 64\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PARAMETER set is not canonical"):
        validate_minicpm_modelfile(changed)


def test_modelfile_contract_rejects_duplicate_from(tmp_path):
    changed = tmp_path / "Modelfile"
    changed.write_text(
        DEFAULT_MODELFILE.read_text(encoding="utf-8")
        + "\nFROM hf.co/openbmb/MiniCPM5-1B-GGUF:Q4_K_M\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one FROM"):
        validate_minicpm_modelfile(changed)


def test_docker_quickstart_has_a_no_create_identity_verification_step():
    text = (_REPO / "docs" / "docker_quickstart.md").read_text(encoding="utf-8")
    assert "tools.setup_minicpm --verify-only" in text
