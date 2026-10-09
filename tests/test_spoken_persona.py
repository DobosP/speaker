"""Opt-in prompt and registry-backed summaries retain truth and task boundaries."""
from threading import Event

import pytest

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult, CapabilitySpec
from always_on_agent.models import CLOUD_EGRESS_SCOPE_CONTEXT_KEY, CloudEgressScope
from core.capabilities import attach_llm_capabilities
from core.persona import PersonaConfig, build_system_prompt, capability_summary_for_query


class Model:
    def __init__(self):
        self.calls = []
    def stream(self, prompt, **kwargs):
        self.calls.append(prompt)
        yield "A useful answer."
    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return "A useful answer."


def registry():
    reg = CapabilityRegistry()
    for name, summary, egress, visible in (
        ("assistant.answer", "Answer questions and tell stories", "local", True),
        ("vault.search", "Search configured local notes", "local", True),
        ("web.search", "Search the web", "cloud", True),
        ("internal.audit", "Never advertise this", "local", False),
    ):
        reg.register(name, lambda *a: CapabilityResult(True, "unused"),
                     spec=CapabilitySpec(name=name, summary=summary, egress=egress, user_facing=visible))
    return reg


def persona():
    return PersonaConfig(name="Iris", prompt_profile="spoken")


def test_spoken_prompt_preserves_custom_persona_extra_and_markup_without_registry_block():
    custom = PersonaConfig(name="Iris", persona="Speak kindly.", extra="Use Celsius.", prompt_profile="spoken")
    prompt = build_system_prompt(registry(), persona=custom, markup_guidance="Use [pause] for a pause.")
    assert all(text in prompt for text in ("You are Iris", "Speak kindly.", "Use Celsius.", "Use [pause]"))
    assert "tell requested stories" not in prompt  # no rewritten user input
    assert "give requested stories and explanations in full" in prompt
    assert "no web access" in prompt
    assert "For reference" not in prompt and "Search configured local notes" not in prompt
    assert "no web access" not in build_system_prompt(registry(), persona=custom, web_enabled=True)


@pytest.mark.parametrize("bad", ["compact", "", None, True, [], {}])
def test_prompt_profile_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        PersonaConfig.from_dict({"prompt_profile": bad})


def test_summary_lists_every_visible_available_capability_and_respects_egress():
    reg = registry()
    local = capability_summary_for_query("Iris, what can you do?", reg, persona=persona(), web_enabled=False)
    assert "I'm Iris" in local
    assert "Answer questions and tell stories" in local
    assert "Search configured local notes" in local
    assert "Search the web" not in local and "Never advertise" not in local
    online = capability_summary_for_query("please list your available tools", reg, persona=persona(), web_enabled=True)
    assert "Search the web" in online


@pytest.mark.parametrize("query", ['Translate "what can you do" into French', 'What can you do and open my notes?',
                                    'Explain why people ask what can you do', 'Say "what can you do"',
                                    'What can you do? Then tell me a story.', 'What can you do for my tax problem?'])
def test_only_complete_capability_question_is_intercepted(query):
    assert capability_summary_for_query(query, registry(), persona=persona(), web_enabled=True) is None
    assert capability_summary_for_query("what can you do", registry(), persona=PersonaConfig(), web_enabled=True) is None


def test_summary_avoids_model_and_uses_existing_speech_receipt_path_without_invoking_tools():
    reg, model = registry(), Model()
    attach_llm_capabilities(reg, model, persona=persona())
    heard = []
    result = reg.invoke("assistant.answer", "what are your capabilities?",
                        {"emit_speech": heard.append, CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.LOCAL_ONLY})
    assert result.ok and result.data["handled_local"] and result.data["streamed"]
    assert heard == [result.text] and not model.calls
    assert "Search the web" not in result.text
    # A quote/compound remains a normal answer request.
    reg.invoke("assistant.answer", 'Explain "what can you do"', {})
    assert model.calls


def test_cancelled_summary_never_emits_and_preserves_revocation_metadata():
    reg, model, event = registry(), Model(), Event()
    attach_llm_capabilities(reg, model, persona=persona())
    event.set()
    emitted = []
    result = reg.invoke("assistant.answer", "what can you do", {"cancel_event": event, "emit_speech": emitted.append})
    assert result.ok and result.data["cancelled"]
    assert emitted == [] and model.calls == []


def test_unattested_scope_cannot_advertise_cloud_tools_or_invoke_virtual_equality():
    class HostileScope:
        def __eq__(self, other):
            raise AssertionError("scope equality must not run")
    for scope in ("current_turn_only", HostileScope()):
        reg, model = registry(), Model()
        attach_llm_capabilities(reg, model, persona=persona())
        result = reg.invoke("assistant.answer", "what can you do",
                            {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: scope})
        assert result.ok and "Search the web" not in result.text
        assert not model.calls
