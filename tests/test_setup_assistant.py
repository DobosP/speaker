"""Headless tests for the optional assistant capability setup transaction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

import tools.setup_assistant as assistant
from tools.setup_assistant import (
    SetupError,
    SetupRequest,
    TrustedApp,
    apply_setup,
    main,
    parse_trusted_app,
)


def _run(config: Path, *args: str) -> int:
    return main(["--config", str(config), *args])


def test_no_capability_options_do_not_create_or_read_config(tmp_path, monkeypatch):
    config = tmp_path / "config.local.json"

    def fail_open(*_args, **_kwargs):
        raise AssertionError("no-options setup must not touch the config")

    monkeypatch.setattr(assistant, "_open_parent", fail_open)
    assert _run(config) == 0
    assert not config.exists()


def test_setup_merges_all_capabilities_and_publishes_mode_0600(tmp_path):
    vault = tmp_path / "paul-brain"
    vault.mkdir()
    config = tmp_path / "config.local.json"
    config.write_text(
        json.dumps(
            {
                "sherpa": {"asr_model": "/models/asr", "nested": {"keep": 1}},
                "obsidian": {"max_results": 7},
                "reminders": {"timezone": "Europe/Bucharest"},
                "trusted_apps": {"policy": "exact-id", "apps": {}},
                "unrelated": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o644)

    rc = _run(
        config,
        "--obsidian-vault",
        str(vault),
        "--enable-reminders",
        "--trust-app",
        "obsidian=md.obsidian.Obsidian.desktop",
    )

    assert rc == 0
    published = json.loads(config.read_text(encoding="utf-8"))
    assert published["sherpa"] == {
        "asr_model": "/models/asr",
        "nested": {"keep": 1},
    }
    assert published["unrelated"] == [1, 2, 3]
    assert published["obsidian"] == {
        "enabled": True,
        "max_results": 7,
        "vault_root": str(vault.resolve()),
    }
    assert published["reminders"] == {
        "enabled": True,
        "timezone": "Europe/Bucharest",
    }
    assert published["trusted_apps"] == {
        "enabled": True,
        "policy": "exact-id",
        "apps": {
            "obsidian": {
                "connector": "desktop_launch",
                "desktop_id": "md.obsidian.Obsidian.desktop",
                "operations": ["open"],
            }
        },
    }
    assert stat.S_IMODE(config.stat().st_mode) == 0o600


def test_vault_validation_never_enumerates_or_reads_notes(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "private.md").write_text("must not be read", encoding="utf-8")
    config = tmp_path / "config.local.json"

    def fail_iterdir(_self):
        raise AssertionError("vault contents must not be enumerated")

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)
    assert _run(config, "--obsidian-vault", str(vault)) == 0


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_vault_does_not_clobber_existing_config(tmp_path, kind):
    config = tmp_path / "config.local.json"
    original = b'{"keep": true}\n'
    config.write_bytes(original)
    vault = tmp_path / "vault"
    if kind == "file":
        vault.write_text("not a directory", encoding="utf-8")

    assert _run(config, "--obsidian-vault", str(vault)) == 1
    assert config.read_bytes() == original


@pytest.mark.parametrize(
    "value",
    [
        "obsidian=/usr/share/applications/obsidian.desktop",
        "obsidian=../obsidian.desktop",
        "obsidian=obsidian;touch.desktop",
        "obsidian=obsidian desktop",
        "obsidian=obsidian.desktop --evil",
        "UPPER=obsidian.desktop",
        "obsidian",
        "obsidian=a=b.desktop",
    ],
)
def test_trusted_app_rejects_paths_commands_and_ambiguous_values(value):
    with pytest.raises(SetupError):
        parse_trusted_app(value)


def test_trusted_app_parser_accepts_exact_desktop_id():
    assert parse_trusted_app("browser=org.mozilla.firefox.desktop") == TrustedApp(
        alias="browser",
        desktop_id="org.mozilla.firefox.desktop",
    )


def test_disable_preserves_settings_and_untrusts_only_requested_alias(tmp_path):
    config = tmp_path / "config.local.json"
    config.write_text(
        json.dumps(
            {
                "obsidian": {"enabled": True, "vault_root": "/vault", "max_results": 5},
                "reminders": {"enabled": True, "timezone": "local"},
                "trusted_apps": {
                    "enabled": True,
                    "apps": {
                        "browser": {"desktop_id": "browser.desktop"},
                        "notes": {"desktop_id": "notes.desktop"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run(
        config,
        "--disable-obsidian",
        "--disable-reminders",
        "--untrust-app",
        "notes",
    ) == 0
    published = json.loads(config.read_text(encoding="utf-8"))
    assert published["obsidian"] == {
        "enabled": False,
        "vault_root": "/vault",
        "max_results": 5,
    }
    assert published["reminders"] == {"enabled": False, "timezone": "local"}
    assert published["trusted_apps"]["enabled"] is True
    assert set(published["trusted_apps"]["apps"]) == {"browser"}


def test_untrusting_last_app_disables_trusted_apps(tmp_path):
    config = tmp_path / "config.local.json"
    config.write_text(
        json.dumps(
            {
                "trusted_apps": {
                    "enabled": True,
                    "apps": {"notes": {"desktop_id": "notes.desktop"}},
                }
            }
        ),
        encoding="utf-8",
    )

    assert _run(config, "--untrust-app", "notes") == 0
    block = json.loads(config.read_text(encoding="utf-8"))["trusted_apps"]
    assert block == {"enabled": False, "apps": {}}


def test_untrust_only_preserves_disabled_state_when_other_apps_remain(tmp_path):
    config = tmp_path / "config.local.json"
    config.write_text(
        json.dumps(
            {
                "trusted_apps": {
                    "enabled": False,
                    "apps": {
                        "browser": {"desktop_id": "browser.desktop"},
                        "notes": {"desktop_id": "notes.desktop"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert _run(config, "--untrust-app", "notes") == 0
    block = json.loads(config.read_text(encoding="utf-8"))["trusted_apps"]
    assert block["enabled"] is False
    assert set(block["apps"]) == {"browser"}


def test_apply_setup_rejects_enable_disable_conflict_without_mutation():
    original = {"obsidian": {"enabled": True, "vault_root": "/keep"}}
    request = SetupRequest(obsidian_vault="/other", disable_obsidian=True)

    with pytest.raises(SetupError):
        apply_setup(original, request, validated_vault="/other")
    assert original == {"obsidian": {"enabled": True, "vault_root": "/keep"}}


@pytest.mark.parametrize(
    "args",
    [
        ("--trust-app", "notes=notes.desktop", "--trust-app", "notes=other.desktop"),
        ("--untrust-app", "notes", "--untrust-app", "notes"),
        ("--trust-app", "notes=notes.desktop", "--untrust-app", "notes"),
    ],
)
def test_duplicate_or_conflicting_app_edits_are_cli_errors(tmp_path, args):
    config = tmp_path / "config.local.json"
    with pytest.raises(SystemExit) as exc:
        _run(config, *args)
    assert exc.value.code == 2
    assert not config.exists()


def test_malformed_or_non_object_json_is_preserved(tmp_path):
    for index, original in enumerate((b"{not-json\n", b"[1, 2, 3]\n")):
        config = tmp_path / f"config-{index}.json"
        config.write_bytes(original)
        assert _run(config, "--enable-reminders") == 1
        assert config.read_bytes() == original


def test_non_object_touched_section_is_preserved_on_failure(tmp_path):
    config = tmp_path / "config.local.json"
    original = b'{"reminders": ["unexpected"]}\n'
    config.write_bytes(original)

    assert _run(config, "--enable-reminders") == 1
    assert config.read_bytes() == original


def test_symlink_config_target_is_refused_without_touching_referent(tmp_path):
    referent = tmp_path / "real-config.json"
    original = b'{"keep": "referent"}\n'
    referent.write_bytes(original)
    config = tmp_path / "config.local.json"
    config.symlink_to(referent)

    assert _run(config, "--enable-reminders") == 1
    assert config.is_symlink()
    assert referent.read_bytes() == original


def test_symlink_parent_is_refused(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    config = linked_parent / "config.local.json"

    assert _run(config, "--enable-reminders") == 1
    assert not (real_parent / "config.local.json").exists()


def test_atomic_replace_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    config = tmp_path / "config.local.json"
    original = b'{"keep": true}\n'
    config.write_bytes(original)

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(assistant.os, "replace", fail_replace)
    assert _run(config, "--enable-reminders") == 1
    assert config.read_bytes() == original
    assert list(tmp_path.glob(".config.local.json.*.tmp")) == []


def test_concurrent_config_change_is_not_clobbered(tmp_path, monkeypatch):
    config = tmp_path / "config.local.json"
    config.write_text('{"old": true}\n', encoding="utf-8")
    original_reader = assistant._read_existing
    calls = 0

    def racing_reader(parent_fd, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            config.write_text('{"concurrent": true, "padding": "changed"}\n', encoding="utf-8")
        return original_reader(parent_fd, name)

    monkeypatch.setattr(assistant, "_read_existing", racing_reader)
    assert _run(config, "--enable-reminders") == 1
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "concurrent": True,
        "padding": "changed",
    }
    assert list(tmp_path.glob(".config.local.json.*.tmp")) == []


def test_new_config_publication_does_not_depend_on_process_umask(tmp_path):
    config = tmp_path / "config.local.json"
    previous = os.umask(0)
    try:
        assert _run(config, "--enable-reminders") == 0
    finally:
        os.umask(previous)
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
