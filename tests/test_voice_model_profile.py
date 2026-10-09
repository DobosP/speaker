"""Pure selection/native identity contracts; no Ollama/model/audio calls."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from core import voice_model_profile as profile
from core.llm_factory import build_llms
from core.readiness import check_ollama


def config():
    return {"llm": {"backend": "ollama", "main_model": "vision-large", "fast_model": "old-small",
                    "options": {"num_ctx": 8192, "num_predict": 512}, "keep_alive": -1},
            "assistant": {"name": "Iris", "persona": "Be kind.", "extra": "Use Celsius."},
            "privacy": {"local_only": True}, "authority": {"enabled": False}}


def shown():
    return json.loads((Path(__file__).parent / "fixtures/qwen_voice_model_show.json").read_text())


def test_current_is_unchanged_and_explicit_rollback_selection_sticks():
    original = config()
    selected, meta = profile.apply_voice_model_profile(original)
    assert selected is original and meta is None
    base = {**original, "voice_model_profile": "qwen2.5-1.5b"}
    rolled, meta = profile.apply_voice_model_profile(base, "current")
    assert rolled["llm"] == original["llm"]
    assert rolled["voice_model_profile"] == "current" and meta is None
    assert profile.apply_voice_model_profile(rolled)[0] is rolled


def test_profile_is_atomic_idempotent_and_never_reduces_caps_or_capabilities():
    original = config()
    snapshot = deepcopy(original)
    selected, meta = profile.apply_voice_model_profile(original, "qwen2.5-1.5b")
    again, again_meta = profile.apply_voice_model_profile(selected)
    assert original == snapshot
    assert selected == again and meta == again_meta
    assert selected["llm"]["main_model"] == original["llm"]["main_model"]
    assert selected["llm"]["options"] == {"num_ctx": 8192, "num_predict": 512}
    assert selected["llm"]["fast_model"] == profile.PROFILE_ALIAS
    assert selected["llm"]["fast_options"] == profile.PROFILE_FAST_OPTIONS
    assert selected["assistant"] == {**original["assistant"], "prompt_profile": "spoken"}
    assert selected["privacy"] == original["privacy"]
    assert selected["authority"] == original["authority"]
    assert len(meta.sha256) == 64 and meta.schema_version == 1


@pytest.mark.parametrize("bad", ["", "Qwen", None, False, [], {}])
def test_profile_names_are_strict(bad):
    base = config()
    base["voice_model_profile"] = bad
    with pytest.raises(ValueError):
        profile.apply_voice_model_profile(base)


def test_profile_refuses_incompatible_native_backend():
    base = config()
    base["llm"]["backend"] = "llamacpp"
    with pytest.raises(ValueError, match="ollama"):
        profile.apply_voice_model_profile(base, "qwen2.5-1.5b")


def test_actual_public_import_fixture_has_pinned_portable_identity():
    actual = shown()
    assert profile.verify_voice_model_identity(show=lambda _: actual).ok
    # A Windows/other-user cache changes only FROM pathname, not native config.
    actual["modelfile"] = actual["modelfile"].replace("/model-cache/", "C:\\Users\\Public\\models\\")
    assert profile.verify_voice_model_identity(show=lambda _: actual).ok


@pytest.mark.parametrize("change", ["blob", "template", "parameters", "system", "capabilities", "family", "precision"])
def test_exact_native_contract_rejects_mutated_effective_config(change):
    actual = shown()
    if change == "blob":
        actual["modelfile"] = actual["modelfile"].replace(profile.PROFILE_RUNTIME_SHA256, "f" * 64)
    elif change == "template":
        actual["modelfile"] = actual["modelfile"].replace("<|im_start|>", "<|broken|>")
        actual["template"] += "unsafe prefix"
    elif change == "parameters":
        actual["parameters"] += "\nnum_ctx 1024"
    elif change == "system":
        actual["modelfile"] += "\nSYSTEM arbitrary instructions"
    elif change == "capabilities":
        actual["capabilities"].append("tools")
    elif change == "family":
        actual["details"]["family"] = "llama"
    else:
        actual["details"]["quantization_level"] = "Q2_K"
    identity = profile.verify_voice_model_identity(show=lambda _: actual)
    assert not identity.ok and identity.error == "voice_model_identity_mismatch"


def test_readiness_requires_native_identity_and_has_correct_setup_hint():
    row = check_ollama([profile.PROFILE_ALIAS], lister=lambda: [profile.PROFILE_ALIAS], show=lambda _: shown())[-1]
    assert row.ok
    row = check_ollama([profile.PROFILE_ALIAS], lister=lambda: [])[-1]
    assert not row.ok and "tools.setup_voice_model" in row.hint
    row = check_ollama([profile.PROFILE_ALIAS], lister=lambda: [profile.PROFILE_ALIAS],
                       show=lambda _: (_ for _ in ()).throw(RuntimeError("private backend detail")))[-1]
    assert not row.ok and row.detail == "voice_model_identity_unavailable"


def capture_factory(monkeypatch):
    clients = []
    class Model:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            clients.append(self)
    monkeypatch.setattr("core.llm_factory.OllamaLLM", Model)
    return clients


def args(**kwargs):
    return Namespace(llm="ollama", model=None, fast_model=None, **kwargs)


def test_direct_factory_auto_selects_profile_and_keeps_answer_caps(monkeypatch):
    capture_factory(monkeypatch)
    base = config()
    base["voice_model_profile"] = "qwen2.5-1.5b"
    main, fast = build_llms(args(), base)
    assert main.model == "vision-large" and fast.model == profile.PROFILE_ALIAS
    assert main.options == {"num_ctx": 8192, "num_predict": 512}
    assert fast.options == {**main.options, **profile.PROFILE_FAST_OPTIONS}
    assert base["llm"]["fast_model"] == "old-small"
    with pytest.raises(ValueError, match="conflicts"):
        build_llms(Namespace(llm="ollama", model=None, fast_model="other"), base)


def test_shared_alias_role_calls_keep_same_native_runner_threads_and_residency(monkeypatch):
    capture_factory(monkeypatch)
    base = config()
    base["llm"].update(main_model="shared", fast_model="shared", main_keep_alive="60s",
                       fast_keep_alive=-1, fast_options={"num_thread": 2, "temperature": 0.0})
    main, fast = build_llms(args(), base)
    assert main.options["num_thread"] == fast.options["num_thread"] == 2
    assert main.keep_alive == fast.keep_alive == -1
    assert "temperature" not in main.options and fast.options["temperature"] == 0
    assert main.options["num_ctx"] == fast.options["num_ctx"] == 8192
    assert "num_thread" not in base["llm"]["options"]


@pytest.mark.parametrize("bad", [{"num_ctx": 1}, {"num_predict": 1}, {"num_thread": True},
                                  {"num_thread": 9}, {"seed": -1}, {"temperature": float("nan")},
                                  {"top_p": 0}, [], None])
def test_role_options_are_bounded_sampling_only(monkeypatch, bad):
    clients = capture_factory(monkeypatch)
    base = config()
    base["llm"]["fast_options"] = bad
    with pytest.raises(ValueError):
        build_llms(args(), base)
    assert not clients


def test_ollama_parameter_map_render_order_and_capability_order_are_not_identity_changes():
    actual = shown()
    lines = actual["modelfile"].splitlines()
    parameters = [line for line in lines if line.startswith("PARAMETER ")]
    actual["modelfile"] = "\n".join(line for line in lines if not line.startswith("PARAMETER ")) + "\n" + "\n".join(reversed(parameters)) + "\n"
    actual["capabilities"].reverse()
    assert profile.verify_voice_model_identity(show=lambda _: actual).ok
    actual["modelfile"] += "PARAMETER temperature 0.0\n"
    assert not profile.verify_voice_model_identity(show=lambda _: actual).ok


def test_duplicate_template_cannot_hide_a_later_effective_prompt_override():
    actual = shown()
    actual["modelfile"] += "\nTEMPLATE different later behavior\n"
    identity = profile.verify_voice_model_identity(show=lambda _: actual)
    assert not identity.ok and identity.error == "voice_model_identity_mismatch"
