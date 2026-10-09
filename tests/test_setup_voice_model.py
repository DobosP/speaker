"""Synthetic bytes/fake transport prove bounded pinned installer behavior."""
from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

from tools import setup_voice_model as setup
from core.voice_model_profile import PROFILE_ALIAS


DATA = b"public synthetic GGUF bytes"


class Opener:
    def __init__(self, data=DATA):
        self.data, self.calls = data, []
    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        return io.BytesIO(self.data)


def pin(monkeypatch):
    monkeypatch.setattr(setup, "PROFILE_SOURCE_BYTES", len(DATA))
    monkeypatch.setattr(setup, "PROFILE_SOURCE_SHA256", sha256(DATA).hexdigest())


def test_exact_bytes_publish_exclusively_and_cached_repeat_does_not_download(monkeypatch, tmp_path):
    pin(monkeypatch)
    opener = Opener()
    asset = setup.ensure_asset(tmp_path, opener=opener)
    assert asset.read_bytes() == DATA
    assert opener.calls[0][1] == 30
    assert setup.PROFILE_REVISION in opener.calls[0][0].full_url
    assert not asset.with_name(setup.PROFILE_FILENAME + ".atomic-part").exists()
    assert setup.ensure_asset(tmp_path, opener=opener) == asset
    assert len(opener.calls) == 1


@pytest.mark.parametrize("data", [b"short", DATA + b"oversized", b"x" * len(DATA)])
def test_bad_size_or_digest_never_publishes_and_cleans_only_owned_partial(monkeypatch, tmp_path, data):
    pin(monkeypatch)
    with pytest.raises(setup.SetupRefused):
        setup.ensure_asset(tmp_path, opener=Opener(data))
    asset = tmp_path / setup.PROFILE_ASSET_RELATIVE
    assert not asset.exists()
    assert not asset.with_name(setup.PROFILE_FILENAME + ".atomic-part").exists()


def test_preexisting_wrong_asset_is_preserved_without_network(monkeypatch, tmp_path):
    pin(monkeypatch)
    asset = tmp_path / setup.PROFILE_ASSET_RELATIVE
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"preserve me")
    opener = Opener()
    with pytest.raises(setup.SetupRefused, match="existing_asset_mismatch"):
        setup.ensure_asset(tmp_path, opener=opener)
    assert asset.read_bytes() == b"preserve me" and not opener.calls


def test_symlink_asset_cannot_be_used_even_when_target_bytes_match(monkeypatch, tmp_path):
    pin(monkeypatch)
    target = tmp_path / "target"
    target.write_bytes(DATA)
    asset = tmp_path / setup.PROFILE_ASSET_RELATIVE
    asset.parent.mkdir(parents=True)
    asset.symlink_to(target)
    with pytest.raises(setup.SetupRefused, match="existing_asset_mismatch"):
        setup.ensure_asset(tmp_path, opener=Opener())
    assert target.read_bytes() == DATA


def test_parent_symlink_and_unowned_partial_are_refused_without_replacing(monkeypatch, tmp_path):
    pin(monkeypatch)
    actual = tmp_path / "other"
    actual.mkdir()
    (tmp_path / "models").symlink_to(actual, target_is_directory=True)
    with pytest.raises(setup.SetupRefused, match="parent_symlink"):
        setup.ensure_asset(tmp_path, opener=Opener())
    (tmp_path / "models").unlink()
    asset = tmp_path / setup.PROFILE_ASSET_RELATIVE
    asset.parent.mkdir(parents=True)
    partial = asset.with_name(setup.PROFILE_FILENAME + ".atomic-part")
    partial.write_bytes(b"another installer")
    with pytest.raises(FileExistsError):
        setup.ensure_asset(tmp_path, opener=Opener())
    assert partial.read_bytes() == b"another installer"


def test_deadline_is_finite_and_never_publishes(monkeypatch, tmp_path):
    pin(monkeypatch)
    moments = iter([0, 601])
    with pytest.raises(setup.SetupRefused, match="deadline"):
        setup.ensure_asset(tmp_path, opener=Opener(), clock=lambda: next(moments))
    assert not (tmp_path / setup.PROFILE_ASSET_RELATIVE).exists()


@pytest.mark.parametrize("host", ["http://remote:11434", "https://127.0.0.1:11434",
                                  "http://u:p@127.0.0.1:11434", "http://127.0.0.1:11434/x",
                                  "http://127.0.0.1", "http://127.0.0.1:11434?secret=yes"])
def test_installation_requires_explicit_loopback_endpoint(host):
    with pytest.raises(setup.SetupRefused):
        setup.local_host(host)


@pytest.mark.parametrize("url", ["http://huggingface.co/model", "https://evil.example/model",
                                 "https://huggingface.co.evil.example/model", "https://u:p@huggingface.co/model"])
def test_redirect_authority_is_allowlisted(url):
    with pytest.raises(setup.SetupRefused, match="redirect_refused"):
        setup.PublicRedirect().redirect_request(Request("https://huggingface.co/start"), None, 302, "", {}, url)


def test_existing_wrong_alias_is_never_overwritten(monkeypatch, tmp_path):
    pin(monkeypatch)
    asset = tmp_path / "model.gguf"
    asset.write_bytes(DATA)
    client = SimpleNamespace(list=lambda: {"models": [{"model": PROFILE_ALIAS}]}, show=lambda _: {})
    with pytest.raises(setup.SetupRefused, match="existing_alias_mismatch"):
        setup.install_alias(asset, "http://127.0.0.1:11434", client=client,
                            run=lambda *a, **k: pytest.fail("must not replace alias"))


def test_import_is_bounded_sanitized_then_exactly_verified(monkeypatch, tmp_path):
    pin(monkeypatch)
    asset = tmp_path / "model.gguf"
    asset.write_bytes(DATA)
    public_show = json.loads((Path(__file__).parent / "fixtures/qwen_voice_model_show.json").read_text())
    client = SimpleNamespace(list=lambda: {"models": []}, show=lambda _: public_show)
    monkeypatch.setattr(setup.shutil, "which", lambda _: "/public/ollama")
    monkeypatch.setenv("OLLAMA_API_KEY", "private-test-canary")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        text = Path(command[-1]).read_text()
        assert "PARAMETER num_ctx 4096" in text
        assert 'PARAMETER stop "<|im_end|>"' in text
        assert "private-test-canary" not in str(kwargs)
        assert kwargs["timeout"] == 120 and kwargs["stdout"] == kwargs["stderr"] == setup.subprocess.DEVNULL
        return SimpleNamespace(returncode=0)
    setup.install_alias(asset, "http://127.0.0.1:11434", client=client, run=run)
    assert calls[0][0][2] == PROFILE_ALIAS
    assert calls[0][1]["env"]["OLLAMA_HOST"] == "http://127.0.0.1:11434"


def test_cli_endpoint_rejection_never_prints_credentials_or_starts_download(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(setup, "ensure_asset", lambda *a, **k: pytest.fail("must reject before I/O"))
    assert setup.main(["--profile", "qwen2.5-1.5b", "--root", str(tmp_path),
                       "--host", "http://u:PASSWORD_CANARY@127.0.0.1:11434"]) == 2
    captured = capsys.readouterr()
    assert "PASSWORD_CANARY" not in captured.out + captured.err
    assert "local_endpoint_required" in captured.err
