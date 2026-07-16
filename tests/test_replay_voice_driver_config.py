from __future__ import annotations

from tests import replay_voice_driver


def test_recorded_replay_disables_machine_local_tool_providers(monkeypatch):
    import core.config

    source = {
        "device": "desktop",
        "sherpa": {"asr_encoder": "/private/model.onnx"},
        "obsidian": {"enabled": True, "vault_path": "/private/vault"},
        "reminders": {"enabled": True, "store_path": "/private/reminders.db"},
        "trusted_apps": {"enabled": True, "apps": {"notes": "notes.desktop"}},
    }
    monkeypatch.setattr(core.config, "load_config", lambda: source)
    monkeypatch.setattr(
        core.config,
        "apply_device_profile",
        lambda config, _device: config,
    )

    isolated = replay_voice_driver._runtime_config()

    assert isolated["sherpa"] == source["sherpa"]
    assert isolated["obsidian"]["enabled"] is False
    assert isolated["reminders"]["enabled"] is False
    assert isolated["trusted_apps"]["enabled"] is False
    assert source["obsidian"]["enabled"] is True
    assert source["reminders"]["enabled"] is True
    assert source["trusted_apps"]["enabled"] is True
