"""llm-inference-5: the on-device (llamacpp) LLM path is provisionable.

Covers the three pieces that made the phone tier unrunnable as shipped:
  * `setup_models --gguf` fetches the MiniCPM5 GGUF from the shared manifest;
  * the phone/phone_lite profiles point at exactly those filenames;
  * build_llms gives an actionable acquisition hint instead of a dead-end exit;
  * requirements-ondevice.txt declares the runtime dep.

Tier 0: the HF download is injected (no network); everything else is static.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import pytest

_REPO = Path(__file__).resolve().parents[1]


def test_fetch_gguf_models_uses_manifest_coords(tmp_path):
    from tools.bench.models import load_manifest
    from tools.setup_models import GGUF_KEYS, fetch_gguf_models

    manifest = load_manifest(None)
    calls = []

    def fake_download(*, repo_id, filename, local_dir, token, force_download):
        calls.append((repo_id, filename, local_dir))
        p = Path(local_dir) / filename
        p.write_bytes(b"GGUF")
        return str(p)

    out = fetch_gguf_models(manifest, str(tmp_path), token="tok", download=fake_download)

    assert set(out) == set(GGUF_KEYS)
    # Both logical roles resolve to the one public MiniCPM5 Q4 artifact.
    assert out["main_gguf"].endswith("MiniCPM5-1B-Q4_K_M.gguf")
    assert out["fast_gguf"].endswith("MiniCPM5-1B-Q4_K_M.gguf")
    # Files actually landed in the requested dir.
    assert (tmp_path / "MiniCPM5-1B-Q4_K_M.gguf").exists()
    assert len(calls) == 1  # identical main/fast artifact is fetched once


def test_phone_profile_paths_match_gguf_manifest_filenames():
    """setup_models --gguf downloads models/<manifest file>; the phone profiles
    must point llm.main/fast_model_path at exactly those names, or the fetched
    weights won't be found. Pin that they don't drift apart."""
    from tools.bench.models import load_manifest

    cfg = json.loads((_REPO / "config.json").read_text())
    manifest = load_manifest(None)
    for profile in ("phone", "phone_lite"):
        llm = cfg["device_profiles"][profile]["llm"]
        assert Path(llm["main_model_path"]).name in (
            manifest["main_gguf"]["file"],
            manifest["fast_gguf"]["file"],  # phone_lite uses the 1b for both tiers
        )
        assert Path(llm["fast_model_path"]).name == manifest["fast_gguf"]["file"]


def test_requirements_ondevice_declares_llamacpp_runtime():
    txt = (_REPO / "requirements-ondevice.txt").read_text()
    assert "llama-cpp-python" in txt
    assert "huggingface_hub" in txt or "huggingface-hub" in txt


def test_build_llms_llamacpp_missing_path_gives_actionable_hint():
    """The bare SystemExit became a guided acquisition message."""
    from core.llm_factory import build_llms

    config = {"llm": {"backend": "llamacpp"}}  # no main_model_path
    args = SimpleNamespace(llm="llamacpp", model=None, fast_model=None)
    with pytest.raises(SystemExit) as exc:
        build_llms(args, config)
    msg = str(exc.value)
    assert "setup_models --gguf" in msg
    assert "requirements-ondevice.txt" in msg
