"""Tests for the capability manifest + capability-aware system prompt.

The manifest (CapabilitySpec on the registry) is the single source of truth that
drives the ReAct tool catalog, the model's self-description, and the startup
reconciliation -- so none of them can drift from the actually-registered
providers, and the model knows what it is and what skills it has.
"""
from __future__ import annotations

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
    create_default_capabilities,
)
from always_on_agent.react import DEFAULT_TOOLS, ReactPlanner, attach_react_capability

from core.persona import (
    DEFAULT_SYSTEM,
    PersonaConfig,
    build_system_prompt,
    render_skills,
)


def _ok(query, context):
    return CapabilityResult(True, "ok")


# --- registry manifest --------------------------------------------------------


def test_register_with_spec_is_queryable():
    reg = CapabilityRegistry()
    spec = CapabilitySpec("x.do", summary="do the thing", planner_tool=True)
    reg.register("x.do", _ok, spec=spec)
    assert reg.spec("x.do") is spec
    assert reg.manifest() == (spec,)
    assert reg.planner_tools() == ("x.do",)


def test_register_override_without_spec_preserves_metadata():
    # The LLM-backed assistant / §9.7 web.search re-register WITHOUT a spec; the
    # original metadata must survive the swap.
    reg = CapabilityRegistry()
    spec = CapabilitySpec("assistant.answer", summary="answer", user_facing=True)
    reg.register("assistant.answer", _ok, spec=spec)
    reg.register("assistant.answer", lambda q, c: CapabilityResult(True, "llm"))
    assert reg.spec("assistant.answer") is spec
    assert reg.invoke("assistant.answer", "hi").text == "llm"


def test_direct_live_capability_authority_is_strict_and_central():
    calls: list[str] = []
    reg = CapabilityRegistry()
    reg.register(
        "device.open",
        lambda query, context: calls.append(query) or CapabilityResult(True, "opened"),
        spec=CapabilitySpec(
            "device.open",
            "open a trusted app",
            side_effecting=True,
            authority="direct_live",
            requires_confirmation=True,
        ),
    )

    for context in (
        {},
        {"origin": "live_audio"},
        {"origin": "live_audio", "direct_user_instruction": True},
        {
            "origin": "unknown",
            "direct_user_instruction": True,
            "confirmed": True,
        },
        {
            "origin": type("ForgedOrigin", (), {"value": "live_audio"})(),
            "direct_user_instruction": True,
            "confirmed": True,
        },
        {
            "origin": "live_audio",
            "direct_user_instruction": 1,
            "confirmed": True,
        },
        {
            "origin": "live_audio",
            "direct_user_instruction": True,
            "confirmed": "yes",
        },
    ):
        result = reg.invoke("device.open", "notes", context)
        assert result.ok is True
        assert result.data == {"executed": False, "blocked": "action_authority"}
    assert calls == []

    result = reg.invoke(
        "device.open",
        "notes",
        {
            "origin": "live_audio",
            "direct_user_instruction": True,
            "confirmed": True,
        },
    )
    assert result.text == "opened"
    assert calls == ["notes"]


def test_verified_owner_authority_adds_identity_requirement():
    calls: list[str] = []
    reg = CapabilityRegistry()
    reg.register(
        "device.sensitive",
        lambda query, context: calls.append(query) or CapabilityResult(True, "done"),
        spec=CapabilitySpec(
            "device.sensitive",
            "sensitive action",
            side_effecting=True,
            authority="verified_owner",
            requires_confirmation=True,
        ),
    )
    base = {
        "origin": "live_audio",
        "direct_user_instruction": True,
        "confirmed": True,
    }
    assert reg.invoke("device.sensitive", "x", base).data["executed"] is False
    assert calls == []
    assert reg.invoke(
        "device.sensitive", "x", {**base, "owner_verified": True}
    ).text == "done"
    assert calls == ["x"]


def test_capability_authority_schema_rejects_typos_and_unbound_confirmation():
    import pytest

    with pytest.raises(ValueError, match="authority"):
        CapabilitySpec("x", "x", authority="verified-onwer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confirmation"):
        CapabilitySpec("x", "x", requires_confirmation=True)


def test_planner_catalog_excludes_every_side_effecting_capability():
    reg = CapabilityRegistry()
    reg.register(
        "safe.read",
        _ok,
        spec=CapabilitySpec("safe.read", "read", planner_tool=True),
    )
    reg.register(
        "unsafe.write",
        _ok,
        spec=CapabilitySpec(
            "unsafe.write", "write", planner_tool=True, side_effecting=True
        ),
    )
    assert reg.planner_tools() == ("safe.read",)


def test_default_capabilities_all_have_specs():
    reg = create_default_capabilities()
    for name in reg.names():
        assert reg.spec(name) is not None, f"{name} has no CapabilitySpec"


def test_describe_planner_uses_when_to_use():
    reg = CapabilityRegistry()
    reg.register(
        "web.search", _ok,
        spec=CapabilitySpec("web.search", summary="search the web", when_to_use="use for current info"),
    )
    assert "use for current info" in reg.describe(["web.search"], planner=True)
    assert "search the web" in reg.describe(["web.search"], planner=False)
    # An unknown name degrades gracefully instead of vanishing.
    assert "- mystery.tool:" in reg.describe(["mystery.tool"])


# --- ReAct catalog is driven by the manifest (no drift) -----------------------


def test_react_catalog_comes_from_registry_specs():
    reg = create_default_capabilities()
    planner = ReactPlanner(object(), reg, tools=("web.search", "research.local"))
    catalog = planner._catalog()
    # Descriptions are the registry's planner-facing when_to_use, not a hand-typed
    # table that could drift.
    assert reg.spec("web.search").when_to_use in catalog
    assert reg.spec("research.local").when_to_use in catalog


def test_default_planner_tools_match_manifest_planner_flags():
    # The hand-listed DEFAULT_TOOLS must equal the registry's planner_tool set --
    # the exact drift the reconciliation check guards.
    reg = create_default_capabilities()
    assert set(DEFAULT_TOOLS) == set(reg.planner_tools())


def test_attach_react_registers_a_spec():
    reg = create_default_capabilities()

    class _LLM:
        def generate(self, prompt, *, system=None):
            return "FINAL: done"

        def stream(self, prompt, *, system=None):
            yield "FINAL: done"

    attach_react_capability(reg, _LLM())
    assert reg.spec("agent.react") is not None
    assert reg.spec("agent.react").planner_tool is False  # never a tool for itself


# --- capability-aware system prompt -------------------------------------------


def test_default_system_prompt_unchanged_byte_for_byte():
    # Recomposed from parts but must stay identical (pinned elsewhere too).
    assert "generate it yourself" in DEFAULT_SYSTEM
    assert "no web access" in DEFAULT_SYSTEM
    assert DEFAULT_SYSTEM.count("\n") == 0  # still a single paragraph
    # The registry-backed skills block + its enumeration guidance must NEVER leak
    # into the legacy default prompt -- they live only in build_system_prompt.
    assert "For reference, here is what you can do" not in DEFAULT_SYSTEM
    assert "do not claim any ability that is not on it" not in DEFAULT_SYSTEM


def test_build_system_prompt_enumerates_user_facing_skills():
    reg = create_default_capabilities()
    prompt = build_system_prompt(reg, web_enabled=True)
    # User-facing, model-deliverable skills appear...
    assert "answer questions and chat directly from your own knowledge" in prompt
    assert "research a topic and give a recommendation" in prompt
    assert "search the web for current information" in prompt
    # ...internal-only steps (user_facing=False) do not...
    assert "outline the scope and key questions" not in prompt  # research.scope
    # ...and silent, mode-gated side-effects are NOT advertised, or the answering
    # LLM would claim it took a note / ran a command when it only replied in text.
    assert "take dictation" not in prompt
    assert "take meeting notes" not in prompt
    assert "run a system command" not in prompt


def test_build_system_prompt_enumerates_both_local_skills_when_web_off():
    # The reported live bug: with web off the model surfaced assistant.answer but
    # DROPPED research.local. Both user-facing local skills must be present.
    reg = create_default_capabilities()
    prompt = build_system_prompt(reg, web_enabled=False)
    assert "answer questions and chat directly from your own knowledge" in prompt
    assert "research a topic and give a recommendation" in prompt


def test_build_system_prompt_has_faithful_enumeration_guidance():
    # The skills block must carry guidance that (a) keeps "just answer the request"
    # the DEFAULT -- so a small answering model doesn't recite its capabilities on
    # every (esp. garbled) turn -- while (b) describing exactly these capabilities,
    # and inventing nothing, ONLY when asked. Present web on AND off so neither the
    # default-answer nor the anti-confabulation directive can silently regress.
    reg = create_default_capabilities()
    for web in (False, True):
        prompt = build_system_prompt(reg, web_enabled=web)
        # (a) answering the request is the dominant default, gated on an explicit ask
        assert "just answer the user's actual request directly" in prompt
        assert "what you can do" in prompt
        # (b) a hard "don't invent" clause...
        assert "do not claim any ability that is not on it" in prompt
        # ...and the story/poem/joke confabulation is re-homed onto answering,
        # not advertised as a standalone skill.
        assert "not a separate skill" in prompt


def test_side_effecting_skills_in_manifest_but_not_user_facing():
    reg = create_default_capabilities()
    for name in ("dictation.clean", "meeting.note", "command.stage"):
        spec = reg.spec(name)
        assert spec is not None, f"{name} dropped from manifest"
        assert spec.user_facing is False, f"{name} should not be advertised to the model"


def test_build_system_prompt_web_line_is_conditional():
    reg = create_default_capabilities()
    no_web = build_system_prompt(reg, web_enabled=False)
    web = build_system_prompt(reg, web_enabled=True)
    assert "no web access" in no_web
    assert "no web access" not in web
    assert "can search the web" in web


def test_web_skill_omitted_when_web_disabled():
    # Coherence: don't tell the model it can search the web while the limits line
    # says it can't. The cloud-egress skill is gated on web_enabled.
    reg = create_default_capabilities()
    assert "search the web for current information" not in build_system_prompt(reg, web_enabled=False)
    assert "search the web for current information" in build_system_prompt(reg, web_enabled=True)
    # local skills are unaffected
    assert "answer questions and chat directly" in build_system_prompt(reg, web_enabled=False)


def test_react_final_system_uses_persona_name():
    reg = create_default_capabilities()
    assert "You are Aria," in ReactPlanner(object(), reg, persona_name="Aria")._final_system()
    assert "You are Aria," not in ReactPlanner(object(), reg)._final_system()


def test_build_system_prompt_uses_persona_identity():
    reg = create_default_capabilities()
    prompt = build_system_prompt(reg, persona=PersonaConfig(name="Aria", extra="Be warm."))
    assert prompt.startswith("You are Aria, a local, on-device voice assistant.")
    assert "Be warm." in prompt


def test_render_skills_empty_for_stub_registry():
    assert render_skills(object()) == ""
    assert render_skills(CapabilityRegistry()) == ""


def test_persona_config_from_dict_defaults_empty():
    p = PersonaConfig.from_dict(None)
    assert p.name == "" and p.persona == "" and p.extra == ""
    p2 = PersonaConfig.from_dict({"name": "Aria", "persona": "calm and concise"})
    assert p2.name == "Aria" and p2.persona == "calm and concise"


# --- end-to-end: the runtime actually feeds the model its skills --------------


class _RecordingLLM:
    def __init__(self):
        self.systems: list[str] = []

    def generate(self, prompt, *, system=None, images=None):
        self.systems.append(system or "")
        return "ok"

    def stream(self, prompt, *, system=None, images=None):
        self.systems.append(system or "")
        yield "ok"


def test_runtime_feeds_capability_aware_system_to_model():
    from always_on_agent.events import Mode

    from core.engines.scripted import ScriptedEngine
    from core.runtime import VoiceRuntime

    llm = _RecordingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, llm, start_mode=Mode.ASSISTANT, persona=PersonaConfig(name="Aria")
    )
    runtime.start(run_bus=False)
    try:
        engine.final("tell me something")
        assert runtime.wait_idle()
        assert llm.systems, "the model was never invoked"
        system = llm.systems[-1]
        assert "You are Aria, a local, on-device voice assistant." in system
        assert "answer questions and chat directly from your own knowledge" in system
    finally:
        runtime.stop()


def test_runtime_prompt_reflects_web_disabled():
    # End-to-end §9.7 coherence: with web search off (the default), the model is
    # told it has no web access AND the web skill is not advertised.
    from always_on_agent.events import Mode

    from core.engines.scripted import ScriptedEngine
    from core.runtime import VoiceRuntime

    llm = _RecordingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT)  # web disabled by default
    runtime.start(run_bus=False)
    try:
        engine.final("tell me something")
        assert runtime.wait_idle()
        system = llm.systems[-1]
        assert "no web access" in system
        assert "search the web for current information" not in system
    finally:
        runtime.stop()
