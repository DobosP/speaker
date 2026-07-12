"""Pure contract tests for the autonomous memory probe's semantic grader."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from always_on_agent.memory import MemoryItem
from always_on_agent.sqlite_memory import SqliteVecMemory
import core.capabilities as core_capabilities
import tools.autotest.memory_probe as memory_probe
from tools.autotest.memory_probe import (
    _uses_recalled_canary,
    _uses_recalled_fact,
    run_memory_probe,
)
from tools.setup_minicpm import (
    LOCAL_MODEL,
    SOURCE_MODEL,
    SOURCE_MODEL_BLOB_SHA256,
)


@pytest.mark.parametrize(
    "answer",
    (
        "Teal!",
        "It's teal.",
        "It is TEAL",
        "Your favorite color is teal.",
        "You said your favorite color was teal.",
        "Teal, as you told me.",
        "I remember that your favorite color is teal.",
        "Not blue -- teal.",
    ),
)
def test_recalled_fact_grader_accepts_confident_user_or_scalar_answers(answer):
    assert _uses_recalled_fact(answer, "teal")


@pytest.mark.parametrize(
    "answer",
    (
        "",
        "Your favorite color is blue.",
        "Your favorite color is tealish.",
        "I don't know whether your favorite color is teal.",
        "I cannot confirm teal.",
        "I have no idea; teal?",
        "I don't have a favorite color, but teal is nice.",
        "I do not possess personal tastes; teal is a pleasant choice.",
        "As an AI, I have no personal preferences, though teal sounds good.",
        "I'm an assistant without a favorite; teal is attractive.",
        "I can't be certain it is teal.",
        "Maybe teal.",
        "It might be teal.",
        "I think it's teal.",
        "I doubt it's teal.",
        "I'm not sure, perhaps teal.",
        "Your favorite color is not teal.",
        "It isn't teal.",
        "Teal is not your favorite color.",
        "Teal isn't your favorite color.",
        "Teal? No.",
        "My favorite color is teal.",
        "My favorite color: teal.",
        "My favorite color? Teal.",
        "Teal is my favorite color.",
        "I prefer teal.",
        "I would choose teal.",
        "For me, teal.",
    ),
)
def test_recalled_fact_grader_rejects_missing_uncertain_or_misattributed_answers(answer):
    assert not _uses_recalled_fact(answer, "teal")


def test_recalled_fact_grader_supports_multiword_keywords():
    assert _uses_recalled_fact("Your project codename is Blue Heron.", "blue heron")
    assert not _uses_recalled_fact("Your project codename is blue-heron.", "blue heron")


@pytest.mark.parametrize(
    "answer",
    (
        "Amber Finch.",
        "It was Amber Finch.",
        "Your lighthouse project codename was Amber Finch.",
        "Amber Finch was the lighthouse project codename.",
    ),
)
def test_cross_session_canary_grader_accepts_scalar_or_grounded_subject(answer):
    assert _uses_recalled_canary(
        answer, "Amber Finch", ("lighthouse", "project", "codename")
    )


@pytest.mark.parametrize(
    "answer",
    (
        "The assistant's lighthouse project codename is Amber Finch.",
        "Another project codename is Amber Finch.",
        "Amber Finch is a nice bird.",
        "The lighthouse project codename is Blue Heron.",
        "Amber Finch is a nice bird. The lighthouse project codename is Blue Heron.",
        "Amber Finch is a nice bird, and the lighthouse project codename is Blue Heron.",
        "Amber Finch is a bird with a lighthouse project whose codename is Blue Heron.",
        "The lighthouse project codename was Amber Finch, but actually it was Blue Heron.",
        "The lighthouse project codename was Blue Heron; Amber Finch is a bird.",
        "The lighthouse project codename was Amber Finch or Blue Heron.",
        "The lighthouse project codename was Amber Finch. Correction: Blue Heron.",
        "Amber Finch was not the lighthouse project codename; Blue Heron was.",
        "The statement that the lighthouse project codename was Amber Finch is false.",
        "The lighthouse project codename was Amber Finch, which is incorrect.",
        "The lighthouse project codename was Amber Finch, not really.",
        "It is not true that the lighthouse project codename was Amber Finch.",
        "The garden project codename is Amber Finch.",
        "The rival project codename is Amber Finch.",
        "The unrelated project codename is Amber Finch.",
    ),
)
def test_cross_session_canary_grader_rejects_wrong_subject_or_value(answer):
    assert not _uses_recalled_canary(
        answer, "Amber Finch", ("lighthouse", "project", "codename")
    )


def test_memory_probe_reports_fenced_cross_session_main_route_separately():
    result = run_memory_probe(llm_kind="echo")

    assert result.ok is False
    assert result.plumbing_ok is True
    assert result.complete is False
    assert result.outcome == "diagnostic_pass"
    assert result.recall_available is True
    assert result.recall_injected is True
    assert result.recall_fenced is True
    assert result.answer_uses_fact is None
    assert result.answer_model == "echo"
    assert result.answer_route == "main"
    assert result.answer_sensitivity == "private"
    assert result.attempt_order == ("main",)
    assert result.main_selected_first is True
    assert result.topology_valid is True
    assert result.recent_history_clean is True
    assert result.controller_answer is False
    assert result.backend_reopened_with_fresh_registry is True


def test_memory_probe_fails_if_reopened_rows_leak_into_native_history(monkeypatch):
    def leaky_all(self):
        return [
            MemoryItem(text, tags, ts)
            for text, tags, ts, _embedding in self._recent_rows()
        ]

    monkeypatch.setattr(SqliteVecMemory, "all", leaky_all)
    result = run_memory_probe(llm_kind="echo")

    assert result.ok is False
    assert result.plumbing_ok is False
    assert result.outcome == "fail"
    assert result.recent_history_clean is False
    assert result.backend_reopened_with_fresh_registry is False


class _CanaryLLM:
    def __init__(self, *, model, **_kwargs):
        self.model = model

    def stream(self, prompt, *, system=None, images=None, history=None):
        if "what was my lighthouse project codename" in prompt.lower():
            yield "Amber Finch."
        else:
            yield "Okay."

    def generate(self, prompt, *, system=None, images=None, history=None):
        return "".join(
            self.stream(prompt, system=system, images=images, history=history)
        )


class _FailingFastCanaryLLM(_CanaryLLM):
    def stream(self, prompt, *, system=None, images=None, history=None):
        if (
            self.model == LOCAL_MODEL
            and "what was my lighthouse project codename" in prompt.lower()
        ):
            raise RuntimeError("forced fast-tier failure")
        yield from super().stream(
            prompt, system=system, images=images, history=history
        )


def _identity_snapshot(role_models, *, blob_suffix="a", effective_suffix="b"):
    def record(model):
        common = {
            "model": model,
            "required": True,
            "ok": True,
            "effective_config_sha256": effective_suffix * 64,
        }
        if model == LOCAL_MODEL:
            return {
                **common,
                "verification": "minicpm_q8_blob_template_parameters",
                "alias": LOCAL_MODEL,
                "source": SOURCE_MODEL,
                "alias_blob_sha256": SOURCE_MODEL_BLOB_SHA256,
                "source_blob_sha256": SOURCE_MODEL_BLOB_SHA256,
                "blob_match": True,
                "pinned_blob_match": True,
                "q8": True,
                "template_match": True,
                "parameters_match": True,
            }
        return {
            **common,
            "verification": "ollama_blob_effective_config",
            "blob_sha256": blob_suffix * 64,
        }

    return {
        "role_models": dict(role_models),
        "models": {
            model: record(model)
            for model in set(role_models.values())
        },
        "ok": True,
    }


def _contract(role_models):
    configured = {"main": "gemma3:12b", "fast": LOCAL_MODEL}
    return memory_probe._ReleaseContract(
        identity_config={"llm": {"host": "http://127.0.0.1:11434"}},
        configured_roles=configured,
        host="http://127.0.0.1:11434",
        host_safe=True,
        shipped_roles=role_models == configured,
        contract_sha256="9" * 64,
    )


def _install_clean_evidence(monkeypatch):
    monkeypatch.setattr(
        memory_probe,
        "repository_metadata",
        lambda: {"revision": "c" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        memory_probe,
        "_release_contract",
        lambda **kwargs: _contract(kwargs["role_models"]),
    )
    monkeypatch.setattr(
        memory_probe,
        "_capture_model_identity",
        lambda role_models, _config: _identity_snapshot(role_models),
    )


def test_real_memory_probe_requires_grounded_answer_and_distinct_roles(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)

    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )
    assert result.ok is True
    assert result.complete is True
    assert result.outcome == "pass"
    assert result.answer_uses_fact is True
    assert result.topology_valid is True

    invalid = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model=LOCAL_MODEL,
    )
    assert invalid.ok is False
    assert invalid.complete is True
    assert invalid.outcome == "fail"
    assert invalid.topology_valid is False


def test_memory_probe_rejects_fast_to_main_fallback_as_promotion(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _FailingFastCanaryLLM)
    _install_clean_evidence(monkeypatch)
    monkeypatch.setattr(
        core_capabilities,
        "strong_subject_overlap",
        lambda _query, _block: False,
    )

    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )

    assert result.answer_route == "main"
    assert result.attempt_order == ("fast", "main")
    assert result.main_selected_first is False
    assert result.plumbing_ok is False
    assert result.ok is False
    assert result.outcome == "fail"


@pytest.mark.parametrize("mutation", ("dirty_before", "dirty_after", "revision", "drift"))
def test_real_memory_probe_requires_clean_stable_revision(monkeypatch, mutation):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)
    calls = 0

    def repository():
        nonlocal calls
        calls += 1
        value = {"revision": "d" * 40, "dirty": False}
        if mutation == "dirty_before" and calls == 1:
            value["dirty"] = True
        elif mutation == "dirty_after" and calls == 2:
            value["dirty"] = True
        elif mutation == "revision":
            value["revision"] = "not-a-revision"
        elif mutation == "drift" and calls == 2:
            value["revision"] = "e" * 40
        return value

    monkeypatch.setattr(memory_probe, "repository_metadata", repository)

    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )

    assert result.plumbing_ok is True
    assert result.answer_uses_fact is True
    assert result.provenance.repository_stable is False
    assert result.provenance_ok is False
    assert result.outcome == "fail"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "error",
        "blob_drift",
        "config_drift",
        "role_drift",
        "record_model",
        "required",
        "verification",
        "pinned",
    ),
)
def test_real_memory_probe_requires_complete_stable_role_identities(
    monkeypatch, mutation
):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)
    calls = 0

    def evidence(role_models, _config):
        nonlocal calls
        calls += 1
        identity = _identity_snapshot(role_models)
        fast = role_models["fast"]
        if mutation == "missing":
            identity["models"].pop(fast)
            identity["ok"] = False
        elif mutation == "error":
            identity["models"][fast]["ok"] = False
        elif mutation == "blob_drift" and calls == 2:
            identity["models"][fast]["alias_blob_sha256"] = "f" * 64
        elif mutation == "config_drift" and calls == 2:
            identity["models"][fast]["effective_config_sha256"] = "f" * 64
        elif mutation == "role_drift" and calls == 2:
            identity["role_models"] = {**role_models, "fast": "other:test"}
        elif mutation == "record_model":
            identity["models"][fast]["model"] = "other:test"
        elif mutation == "required":
            identity["models"][fast]["required"] = False
        elif mutation == "verification":
            identity["models"][fast]["verification"] = "hash-only"
        elif mutation == "pinned":
            identity["models"][fast]["pinned_blob_match"] = False
        return identity

    monkeypatch.setattr(memory_probe, "_capture_model_identity", evidence)

    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )

    assert result.provenance.model_identity_stable is False
    assert result.provenance_ok is False
    assert result.outcome == "fail"


def test_real_memory_probe_rejects_distinct_but_nonshipped_aliases(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)

    result = run_memory_probe(
        llm_kind="ollama",
        model="minicpm-test",
        main_model="gemma-test",
    )

    assert result.topology_valid is True
    assert result.provenance.shipped_roles is False
    assert result.provenance_ok is False
    assert result.outcome == "fail"


def test_echo_memory_probe_never_collects_release_provenance(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Echo diagnostic must not inspect Git or Ollama")

    monkeypatch.setattr(memory_probe, "repository_metadata", forbidden)
    monkeypatch.setattr(memory_probe, "_release_contract", forbidden)
    monkeypatch.setattr(memory_probe, "_capture_model_identity", forbidden)
    result = run_memory_probe(llm_kind="echo")

    assert result.outcome == "diagnostic_pass"
    assert result.provenance.applicable is False
    assert result.provenance_ok is None


def test_memory_result_serializes_release_evidence(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)
    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )

    serialized = asdict(result)
    provenance = serialized["provenance"]
    assert serialized["provenance_ok"] is True
    assert provenance["contract_stable"] is True
    assert provenance["repository_before"]["revision"] == "c" * 40
    assert len(provenance["contract_sha256"]) == 64
    assert set(provenance["identity"]["before"]["models"]) == {
        LOCAL_MODEL,
        "gemma3:12b",
    }


@pytest.mark.parametrize(
    "host",
    (
        "https://127.0.0.1:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://example.com:11434",
        "http://127.0.0.1:11434/path",
    ),
)
def test_memory_probe_host_contract_falls_back_safe_and_red(host):
    resolved, safe = memory_probe._safe_loopback_host(host)
    assert resolved == "http://127.0.0.1:11434"
    assert safe is False


def test_real_memory_probe_recomputes_and_requires_stable_release_contract(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)
    _install_clean_evidence(monkeypatch)
    calls = 0

    def drifting_contract(**kwargs):
        nonlocal calls
        calls += 1
        value = _contract(kwargs["role_models"])
        if calls == 2:
            return memory_probe._ReleaseContract(
                **{**asdict(value), "contract_sha256": "8" * 64}
            )
        return value

    monkeypatch.setattr(memory_probe, "_release_contract", drifting_contract)
    result = run_memory_probe(
        llm_kind="ollama",
        model=LOCAL_MODEL,
        main_model="gemma3:12b",
    )

    assert calls == 2
    assert result.provenance.contract_stable is False
    assert result.provenance_ok is False
    assert result.outcome == "fail"


def test_memory_probe_has_no_public_evidence_injection_seam():
    import inspect

    assert "evidence_provider" not in inspect.signature(run_memory_probe).parameters
