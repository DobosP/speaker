"""P1a cloud hardening: the outbound-cloud PII redactor (a §9.7 last-line net
independent of the regex sensitivity classifier) + the no-CN-in-default-chains
config guard. Tier-0, no network/model.
"""
from __future__ import annotations

import json
import os

from core.app import _build_cloud_client
from core.llm import OpenAICompatLLM, _redact_messages_for_egress


# --- the redactor ------------------------------------------------------------

def test_redact_messages_scrubs_text_content():
    msgs = [
        {"role": "system", "content": "persona. recalled: card 4111 1111 1111 1111"},
        {"role": "user", "content": "my ssn is 123-45-6789 and key sk-abcdef0123456789ABCDEF"},
    ]
    out = _redact_messages_for_egress(msgs)
    assert "4111" not in out[0]["content"] and "[REDACTED_CARD]" in out[0]["content"]
    assert "123-45-6789" not in out[1]["content"] and "sk-abcdef" not in out[1]["content"]


def test_redact_messages_handles_multimodal_parts():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "ssn 123-45-6789"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]}]
    out = _redact_messages_for_egress(msgs)
    assert "[REDACTED_SSN]" in out[0]["content"][0]["text"]
    assert out[0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"  # image untouched


def test_redact_leaves_ordinary_prose_unchanged():
    msgs = [{"role": "user", "content": "what is a good recipe for dinner tonight"}]
    assert _redact_messages_for_egress(msgs) == msgs


def test_egress_redaction_ignores_disable_kill_switch(monkeypatch):
    # Review hardening (XC1): SPEAKER_DISABLE_REDACT is an operator opt-out for the
    # *durable-record* redactor only. The §9.7 cloud-egress net is mandatory and must
    # still scrub PII even when that env var is set -- disabling local-record
    # scrubbing must never silently send a card/SSN to a third-party cloud.
    monkeypatch.setenv("SPEAKER_DISABLE_REDACT", "1")
    msgs = [{"role": "user", "content": "my ssn is 123-45-6789 card 4111 1111 1111 1111"}]
    out = _redact_messages_for_egress(msgs)
    assert "123-45-6789" not in out[0]["content"] and "[REDACTED_SSN]" in out[0]["content"]
    assert "4111" not in out[0]["content"]


# --- per-instance flag (cloud-only; local endpoint unaffected) ---------------

def test_outbound_redaction_off_by_default_is_byte_identical():
    c = OpenAICompatLLM(model="x")  # default redact_pii_outbound=False (e.g. local llama-server)
    kw = c._create_kwargs("my card is 4111 1111 1111 1111", "sys", None, stream=False)
    assert "4111" in str(kw["messages"])  # NOT redacted -> local/legacy behavior unchanged


def test_outbound_redaction_on_scrubs_create_kwargs():
    c = OpenAICompatLLM(model="x", redact_pii_outbound=True)
    kw = c._create_kwargs("my card is 4111 1111 1111 1111", "sys", None, stream=False)
    assert "4111" not in str(kw["messages"])
    assert "[REDACTED_CARD]" in str(kw["messages"])


# --- factory wiring + config guard ------------------------------------------

def test_build_cloud_client_enables_redaction_by_default(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    preset = {"model": "m", "base_url": "https://api.example.com/v1", "api_key_env": "FAKE_KEY", "host": "US"}
    client = _build_cloud_client("p", preset)
    assert client is not None and client._redact_pii_outbound is True
    # explicit opt-out is honored
    client2 = _build_cloud_client("p", preset, redact_pii_outbound=False)
    assert client2._redact_pii_outbound is False


def test_single_cloud_backcompat_enables_redaction(monkeypatch):
    # The single-cloud (llm.cloud only, no providers/chains) path must ALSO enable
    # the outbound PII scrub by default -- it previously omitted the kwarg.
    from core.app import _wrap_cloud
    from core.llm import HedgeLLM, LLMClient

    class _Local(LLMClient):
        def generate(self, prompt, *, system=None, images=None): return "x"
        def stream(self, prompt, *, system=None, images=None): yield "x"

    monkeypatch.setenv("FAKE_KEY", "x")
    llm_cfg = {"cloud": {"enabled": True, "model": "m", "base_url": "https://api.example.com/v1",
                         "api_key_env": "FAKE_KEY"}}
    wrapped = _wrap_cloud(_Local(), llm_cfg)
    assert isinstance(wrapped, HedgeLLM)
    assert wrapped.clouds[0]._redact_pii_outbound is True


def test_default_cloud_chains_have_no_cn_provider():
    # Regression guard: no default failover chain may list a CN-hosted preset, so a
    # mis-classified personal turn cannot cross to a PRC endpoint even with allow_prc.
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
    llm = cfg.get("llm", {})
    providers = llm.get("cloud_providers", {})

    def _is_cn(name: str) -> bool:
        p = providers.get(name, {})
        host = p.get("host") or (p.get("_pricing_usd_per_mtok", {}) or {}).get("host")
        base = (p.get("base_url") or "").lower()
        return (str(host).upper() == "CN") or "deepseek.com" in base or "moonshot" in base

    for chain, names in (llm.get("cloud_chains", {}) or {}).items():
        if chain.startswith("_") or not isinstance(names, list):
            continue
        cn = [n for n in names if _is_cn(n)]
        assert not cn, f"CN-hosted preset(s) {cn} in default chain {chain!r}"
