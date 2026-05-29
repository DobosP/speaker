"""Tests for the OpenRouter presets + the PRC opt-in gate (P3 step 6).

Locked Decision 2: US-hosted chains are the default; PRC-hosted presets
(host=CN, e.g. DeepSeek/Moonshot) are dropped unless ``llm.cloud.allow_prc``
is set. BR8: the drop is INFO-logged distinctly from the missing-API-key drop.

These exercise ``_build_cloud_client`` / ``_wrap_cloud`` directly + assert the
committed config.json's default public chain is led by a US-hosted provider.
No network, no real keys (env is monkeypatched).
"""
from __future__ import annotations

import json
import os

from core.app import _build_cloud_client, _preset_host, _wrap_cloud
from core.llm import OpenAICompatLLM, SensitivityRouterLLM


_CN_PRESET = {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "profile": "deepseek",
    "_pricing_usd_per_mtok": {"in": 0.14, "out": 0.28, "host": "CN"},
}
_US_PRESET = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "openai/gpt-oss-120b",
    "api_key_env": "OPENROUTER_API_KEY",
    "profile": "openai_compat",
    "host": "US",
}


# --- host detection --------------------------------------------------------


def test_preset_host_reads_top_level_and_pricing_block():
    assert _preset_host(_US_PRESET) == "US"  # top-level host
    assert _preset_host(_CN_PRESET) == "CN"  # nested in _pricing_usd_per_mtok
    assert _preset_host({"model": "m"}) is None  # absent


# --- PRC opt-in gate -------------------------------------------------------


def test_cn_preset_dropped_without_allow_prc(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")  # key present, host is the blocker
    assert _build_cloud_client("deepseek_v4_flash", _CN_PRESET, allow_prc=False) is None


def test_cn_preset_kept_with_allow_prc(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    client = _build_cloud_client("deepseek_v4_flash", _CN_PRESET, allow_prc=True)
    assert isinstance(client, OpenAICompatLLM)
    assert client.model == "deepseek-v4-flash"


def test_us_preset_kept_without_allow_prc(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    client = _build_cloud_client("openrouter_gpt_oss_120b", _US_PRESET, allow_prc=False)
    assert isinstance(client, OpenAICompatLLM)
    assert client.model == "openai/gpt-oss-120b"


def test_prc_drop_logs_distinctly_from_missing_key(monkeypatch, caplog):
    """BR8: the PRC drop emits an INFO log distinct from the missing-key drop so
    a user who forgot allow_prc isn't silently degraded to local."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    with caplog.at_level("INFO", logger="speaker.app"):
        assert _build_cloud_client("deepseek_v4_flash", _CN_PRESET, allow_prc=False) is None
    text = caplog.text.lower()
    assert "prc" in text or "host=cn" in text
    assert "api key" not in text  # NOT the missing-key message


def test_missing_key_drop_logs_distinctly(monkeypatch, caplog):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with caplog.at_level("INFO", logger="speaker.app"):
        assert _build_cloud_client("openrouter_gpt_oss_120b", _US_PRESET) is None
    text = caplog.text.lower()
    assert "api key" in text
    assert "prc" not in text  # NOT the PRC message


# --- never logs the secret value -------------------------------------------


def test_drop_logs_never_print_the_secret(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-token-value")
    monkeypatch.delenv("MISSING_THING", raising=False)
    preset = dict(_US_PRESET, api_key_env="MISSING_THING")
    with caplog.at_level("INFO", logger="speaker.app"):
        _build_cloud_client("p", preset)
    assert "super-secret-token-value" not in caplog.text


# --- _wrap_cloud honours allow_prc end to end ------------------------------


def _multi_provider_cfg(allow_prc):
    return {
        "cloud": {"enabled": True, "strategy": "hedge", "allow_prc": allow_prc},
        "cloud_providers": {
            "us": dict(_US_PRESET),
            "cn": dict(_CN_PRESET),
        },
        "cloud_chains": {"public": ["us", "cn"]},
        "cloud_routing": {"default_chain": "public"},
    }


class _Local:
    def generate(self, prompt, *, system=None, images=None):
        return "local"

    def stream(self, prompt, *, system=None, images=None):
        yield "local"


def test_wrap_cloud_drops_cn_from_chain_without_allow_prc(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    wrapped = _wrap_cloud(_Local(), _multi_provider_cfg(allow_prc=False))
    assert isinstance(wrapped, SensitivityRouterLLM)
    # Only the US provider survives the gate.
    assert len(wrapped.chains["public"].clouds) == 1


def test_wrap_cloud_keeps_cn_in_chain_with_allow_prc(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    wrapped = _wrap_cloud(_Local(), _multi_provider_cfg(allow_prc=True))
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert len(wrapped.chains["public"].clouds) == 2


def test_wrap_cloud_plumbs_timeout_and_max_tokens(monkeypatch):
    """BR1/BR4/BR9: the short cloud timeout + per-turn ceiling reach every
    OpenAICompatLLM built (multi-provider site here)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = {
        "cloud": {
            "enabled": True, "strategy": "hedge",
            "timeout_s": 5, "max_tokens": 512,
        },
        "cloud_providers": {"us": dict(_US_PRESET)},
        "cloud_chains": {"public": ["us"]},
        "cloud_routing": {"default_chain": "public"},
    }
    wrapped = _wrap_cloud(_Local(), cfg)
    client = wrapped.chains["public"].clouds[0]
    assert isinstance(client, OpenAICompatLLM)
    assert client._timeout == 5.0
    assert client._max_tokens == 512


def test_wrap_cloud_single_cloud_path_plumbs_timeout_and_max_tokens(monkeypatch):
    """BR9: the single-cloud back-compat site honours timeout_s + max_tokens too."""
    monkeypatch.setenv("LEGACY_KEY", "k")
    cfg = {
        "cloud": {
            "enabled": True, "strategy": "hedge",
            "base_url": "https://x", "model": "m1", "api_key_env": "LEGACY_KEY",
            "timeout_s": 3, "max_tokens": 256,
        }
    }
    wrapped = _wrap_cloud(_Local(), cfg)
    assert isinstance(wrapped.cloud, OpenAICompatLLM)
    assert wrapped.cloud._timeout == 3.0
    assert wrapped.cloud._max_tokens == 256


# --- committed config.json: default public chain is US-hosted --------------


def _load_config():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        return json.load(fh)


def test_config_has_openrouter_us_presets():
    cfg = _load_config()
    providers = cfg["llm"]["cloud_providers"]
    for name in ("openrouter_gpt_oss_120b", "openrouter_llama_3_3_70b"):
        assert name in providers, f"missing preset {name}"
        preset = providers[name]
        assert preset["base_url"] == "https://openrouter.ai/api/v1"
        assert preset["api_key_env"] == "OPENROUTER_API_KEY"
        assert preset["profile"] == "openai_compat"
        assert _preset_host(preset) == "US"


def test_config_default_public_chain_leads_with_us_gpt_oss(monkeypatch):
    """The default public chain leads with the US-hosted gpt-oss-120b, so with
    allow_prc off the first usable public provider is US, not CN."""
    cfg = _load_config()
    public_chain = cfg["llm"]["cloud_chains"]["public"]
    assert public_chain[0] == "openrouter_gpt_oss_120b"
    lead_preset = cfg["llm"]["cloud_providers"][public_chain[0]]
    assert _preset_host(lead_preset) == "US"


def test_config_cloud_block_has_prc_and_cost_controls():
    cfg = _load_config()
    cloud = cfg["llm"]["cloud"]
    assert cloud["allow_prc"] is False
    assert "max_tokens" in cloud
    assert "timeout_s" in cloud
