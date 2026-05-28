"""Tests for the cloud_providers + cloud_chains config wiring in core/app.py.

_wrap_cloud has two paths -- single-cloud back-compat and the new multi-
provider sensitivity-routed shape. This file covers the new path:
provider registry resolution, missing-API-key handling, chain build, and
the SensitivityRouterLLM hookup.
"""
from __future__ import annotations

import json
import os

import pytest

from core.app import _build_cloud_client, _wrap_cloud
from core.llm import (
    HedgeLLM,
    LLMClient,
    OpenAICompatLLM,
    SensitivityRouterLLM,
    capability_context,
)


class _Stub(LLMClient):
    def generate(self, prompt, *, system=None, images=None) -> str:
        return "stub"

    def stream(self, prompt, *, system=None, images=None):
        yield "stub"


# --- preset resolution -----------------------------------------------------


def test_build_cloud_client_requires_model():
    assert _build_cloud_client("x", {"base_url": "..."}) is None
    assert _build_cloud_client("x", {}) is None


def test_build_cloud_client_skips_when_api_key_env_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    preset = {
        "base_url": "https://x.example/v1",
        "model": "m1",
        "api_key_env": "MISSING_KEY",
    }
    assert _build_cloud_client("p", preset) is None


def test_build_cloud_client_constructs_with_env_key(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "secret")
    client = _build_cloud_client(
        "p", {"base_url": "https://x.example/v1", "model": "m1", "api_key_env": "PRESENT_KEY"}
    )
    assert isinstance(client, OpenAICompatLLM)
    assert client.model == "m1"
    assert client._api_key == "secret"


def test_build_cloud_client_ignores_underscore_keys_in_preset():
    """The preset may carry _comment / _pricing_usd_per_mtok metadata."""
    client = _build_cloud_client(
        "p",
        {
            "_comment": "Cerebras Qwen-235B",
            "_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20},
            "base_url": "https://x.example/v1",
            "model": "m1",
        },
    )
    assert isinstance(client, OpenAICompatLLM)


# --- _wrap_cloud disabled --------------------------------------------------


def test_wrap_cloud_disabled_when_cloud_block_off():
    local = _Stub()
    assert _wrap_cloud(local, {"cloud": {"enabled": False}}) is local


def test_wrap_cloud_disabled_when_strategy_local_only():
    local = _Stub()
    cfg = {
        "cloud": {"enabled": True, "strategy": "local_only"},
        "cloud_providers": {"x": {"base_url": "y", "model": "m"}},
        "cloud_chains": {"private": ["x"]},
    }
    assert _wrap_cloud(local, cfg) is local


# --- multi-provider sensitivity-routed path --------------------------------


def test_wrap_cloud_builds_sensitivity_router_with_per_chain_hedges(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.setenv("KEY_B", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge", "hedge_delay_ms": 0},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
            "b": {"base_url": "https://b", "model": "mb", "api_key_env": "KEY_B"},
        },
        "cloud_chains": {
            "private": ["a", "b"],
            "public": ["b"],
        },
        "cloud_routing": {
            "default_chain": "private",
            "sensitivity_to_chain": {"private": "private", "public": "public"},
        },
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert set(wrapped.chains.keys()) == {"private", "public"}
    assert wrapped.default_chain == "private"
    # Each chain entry is a HedgeLLM whose clouds match the chain order.
    for name, expected_n in [("private", 2), ("public", 1)]:
        chain = wrapped.chains[name]
        assert isinstance(chain, HedgeLLM)
        assert len(chain.clouds) == expected_n


def test_wrap_cloud_drops_providers_with_missing_keys(monkeypatch):
    """If an API key is unset, that provider is silently absent from any
    chain it appears in; the remaining chain still works."""
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.delenv("KEY_B", raising=False)
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
            "b": {"base_url": "https://b", "model": "mb", "api_key_env": "KEY_B"},
        },
        "cloud_chains": {"private": ["a", "b"]},
        "cloud_routing": {"default_chain": "private"},
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    chain = wrapped.chains["private"]
    assert len(chain.clouds) == 1  # only "a" survived


def test_wrap_cloud_returns_local_when_every_chain_is_empty(monkeypatch):
    """All API keys missing -> no usable cloud -> stay fully local."""
    monkeypatch.delenv("KEY_A", raising=False)
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
        },
        "cloud_chains": {"private": ["a"]},
        "cloud_routing": {"default_chain": "private"},
    }
    local = _Stub()
    assert _wrap_cloud(local, cfg) is local


def test_wrap_cloud_default_chain_falls_back_when_not_in_chains(monkeypatch):
    """Config typo: default_chain points to a chain name that isn't built.
    The wrapper should still produce a working SensitivityRouterLLM using
    one of the available chains as the default."""
    monkeypatch.setenv("KEY_A", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {"a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"}},
        "cloud_chains": {"public": ["a"]},
        "cloud_routing": {"default_chain": "private"},  # typo: no "private" chain
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert wrapped.default_chain == "public"


def test_wrap_cloud_ignores_underscore_keys_in_providers_and_chains(monkeypatch):
    """The config carries _comment / _pricing metadata keys -- they must
    not be treated as provider names or chain names."""
    monkeypatch.setenv("KEY_A", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "_comment": "metadata, not a provider",
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
        },
        "cloud_chains": {
            "_comment": "metadata, not a chain",
            "private": ["a"],
        },
        "cloud_routing": {"default_chain": "private"},
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert set(wrapped.chains.keys()) == {"private"}


# --- single-cloud back-compat preserved ------------------------------------


def test_wrap_cloud_falls_back_to_single_cloud_shape_when_no_providers(monkeypatch):
    monkeypatch.setenv("LEGACY_KEY", "k")
    cfg = {
        "cloud": {
            "enabled": True,
            "strategy": "hedge",
            "base_url": "https://x",
            "model": "m1",
            "api_key_env": "LEGACY_KEY",
        }
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, HedgeLLM)
    assert isinstance(wrapped.cloud, OpenAICompatLLM)


# --- config.json sanity ----------------------------------------------------


def test_config_json_has_well_formed_cloud_providers_block():
    """The committed config.json's cloud_providers entries must each have a
    base_url + model + api_key_env so users can fill in keys and go."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    providers = cfg["llm"]["cloud_providers"]
    for name, preset in providers.items():
        if name.startswith("_"):
            continue
        assert "base_url" in preset, f"{name} missing base_url"
        assert "model" in preset, f"{name} missing model"
        assert "api_key_env" in preset, f"{name} missing api_key_env"


def test_config_json_chains_reference_only_defined_providers():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    providers = {n for n in cfg["llm"]["cloud_providers"] if not n.startswith("_")}
    for chain_name, preset_names in cfg["llm"]["cloud_chains"].items():
        if chain_name.startswith("_"):
            continue
        for pname in preset_names:
            assert pname in providers, (
                f"chain {chain_name!r} references undefined provider {pname!r}"
            )


def test_config_json_routing_mapping_targets_defined_chains():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    chains = {n for n in cfg["llm"]["cloud_chains"] if not n.startswith("_")}
    routing = cfg["llm"]["cloud_routing"]
    assert routing["default_chain"] in chains
    for sens, chain_name in routing["sensitivity_to_chain"].items():
        assert chain_name in chains, (
            f"sensitivity {sens!r} -> chain {chain_name!r} which is not defined"
        )
