from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.setup_minicpm import LOCAL_MODEL, SOURCE_MODEL, provision

_REPO = Path(__file__).resolve().parents[1]


def test_provision_pulls_official_gguf_then_creates_alias(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM source\n", encoding="utf-8")
    calls = []

    def run(command, *, check):
        calls.append((command, check))

    provision(modelfile=modelfile, runner=run)

    assert calls == [
        (["ollama", "pull", SOURCE_MODEL], True),
        (["ollama", "create", LOCAL_MODEL, "-f", str(modelfile)], True),
    ]


def test_provision_can_rebuild_alias_without_pull(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM source\n", encoding="utf-8")
    calls = []

    provision(
        modelfile=modelfile,
        pull=False,
        runner=lambda command, *, check: calls.append((command, check)),
    )

    assert calls == [
        (["ollama", "create", LOCAL_MODEL, "-f", str(modelfile)], True)
    ]


def test_provision_fails_before_running_when_modelfile_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        provision(
            modelfile=tmp_path / "missing",
            runner=lambda *_args, **_kwargs: pytest.fail("runner was called"),
        )


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
        # MiniCPM5-1B is text-only.  The main role remains a vision-capable
        # Gemma model, so image/screen turns keep their existing route.
        assert llm["main_model"].startswith("gemma3:")


def test_committed_modelfile_pins_openbmb_template_and_sampling():
    text = (
        _REPO / "deploy" / "ollama" / "Modelfile.minicpm5-1b-q8"
    ).read_text(encoding="utf-8")
    assert f"FROM {SOURCE_MODEL}" in text
    assert "<|im_start|>{{ .Role }}" in text
    assert 'PARAMETER stop "<|im_end|>"' in text
    assert "PARAMETER temperature 0.7" in text
    assert "PARAMETER top_p 0.95" in text
