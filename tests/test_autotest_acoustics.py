from __future__ import annotations

import json
import hashlib
import os
import stat
import tempfile
import types

import pytest

from core.virtual_audio import NativeEchoCancelContract, NativeNodeContract
from tools.autotest.acoustics import DelayAcoustics
from tools.autotest.voice_loop import _engine_args


class _NativeOwner:
    pid = 9001
    module_var = "1"
    module_local_id = "42"
    detail = ""

    def __init__(self, config=""):
        self.config = config
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return True, "native EC owner exited"


@pytest.fixture(autouse=True)
def _fake_native_owner(monkeypatch):
    owners = []

    def start(_self, _config):
        owner = _NativeOwner(_config)
        owners.append(owner)
        return owner

    def attest(_self, _owner, *, config_sha256, timeout_sec=5.0):
        del timeout_sec
        return NativeEchoCancelContract(
            owner_pid=_owner.pid,
            client_id="100",
            client_serial="1000",
            config_sha256=config_sha256,
            module_var=_owner.module_var,
            module_local_id=_owner.module_local_id,
            capture_node=NativeNodeContract("101", "1001"),
            source_node=NativeNodeContract("102", "1002"),
            sink_node=NativeNodeContract("103", "1003"),
            playback_node=NativeNodeContract("104", "1004"),
        )

    monkeypatch.setattr(DelayAcoustics, "_start_native_echo_cancel", start)
    monkeypatch.setattr(DelayAcoustics, "_attest_native_owner", attest)
    monkeypatch.setattr(
        DelayAcoustics, "_native_graph_absent", lambda _self, **_kwargs: True
    )
    return owners


class _Modules:
    def __init__(
        self,
        *,
        leaked_module: bool = False,
        orphan_stream: bool = False,
        orphan_stream_id: bool = False,
        orphan_new_stream: bool = False,
        residual_child_stream: bool = False,
        default_drift: bool = False,
    ):
        self.calls = []
        self.next_id = 70
        self.leaked_module = leaked_module
        self.orphan_stream = orphan_stream
        self.orphan_stream_id = orphan_stream_id
        self.orphan_new_stream = orphan_new_stream
        self.residual_child_stream = residual_child_stream
        self.default_drift = default_drift
        self.unloaded = False
        self.source_output_calls = 0

    def __call__(self, argv, **_kwargs):
        call = list(argv)
        self.calls.append(call)
        stdout = ""
        if call[1:] == ["get-default-source"]:
            stdout = (
                "alsa_input.changed\n"
                if self.default_drift and self.next_id > 70
                else "alsa_input.physical\n"
            )
        elif call[1:] == ["get-default-sink"]:
            stdout = "alsa_output.physical\n"
        elif call[1:] == ["list", "short", "modules"] and self.leaked_module:
            stdout = "70\tmodule-null-sink\tsink_name=leaked\n"
        elif call[1:] == ["list", "source-outputs"]:
            self.source_output_calls += 1
            if not self.unloaded:
                stdout = (
                    "Source Output #90\n\tSource: 1\n\tOwner Module: 72\n"
                )
                if self.residual_child_stream and self.source_output_calls >= 2:
                    stdout += (
                        "\n\nSource Output #199\n"
                        "\tSource: 999\n"
                        "\tOwner Module: n/a\n"
                        "\tProperties:\n"
                        '\t\tnode.name = "alsa_capture.python"\n'
                    )
            elif self.orphan_stream:
                stdout = (
                    "Source Output #99\n"
                    "\tSource: 999\n"
                    "\tOwner Module: 72\n"
                    "\tProperties:\n"
                    '\t\tnode.name = "generic.loopback"\n'
                )
            elif self.orphan_stream_id:
                stdout = (
                    "Source Output #90\n"
                    "\tSource: 999\n"
                    "\tOwner Module: n/a\n"
                    "\tProperties:\n"
                    '\t\tnode.name = "generic.loopback"\n'
                )
            elif self.orphan_new_stream:
                stdout = (
                    "Source Output #199\n"
                    "\tSource: 999\n"
                    "\tOwner Module: n/a\n"
                    "\tProperties:\n"
                    '\t\tnode.name = "generic.capture"\n'
                )
        elif call[1:] == ["list", "sink-inputs"] and not self.unloaded:
            stdout = (
                "Sink Input #92\n\tSink: 1\n\tOwner Module: 72\n"
            )
        elif call[1:3] == ["load-module", "module-null-sink"] or (
            len(call) > 2
            and call[1] == "load-module"
            and call[2] == "module-loopback"
        ):
            stdout = str(self.next_id)
            self.next_id += 1
        elif len(call) > 1 and call[1] == "unload-module":
            self.unloaded = True
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_delay_session_owns_exact_virtual_ec_graph_and_cleans_up(
    monkeypatch, _fake_native_owner
):
    modules = _Modules()
    monkeypatch.setattr("tools.autotest.acoustics.subprocess.run", modules)
    ac = DelayAcoustics(prefix="cc_autotest_012345abcdef")

    assert ac.inject_gain == 63
    assert ac.barge_inject_gain == 130
    assert ac.barge_lead_in_ms == 300
    assert (ac.inject_gain / 100.0) ** 3 == pytest.approx(0.25, abs=0.001)

    with ac.session():
        path = ac.contract_path
        assert path is not None and os.path.isfile(path)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["schema"] == "speaker.autotest.virtual-audio/v2"
        assert payload["version"] == 2
        assert payload["parent_pid"] == os.getpid()
        assert payload["default_source"] == "alsa_input.physical"
        assert payload["default_sink"] == "alsa_output.physical"
        assert payload["modules"] == {
            "far": "70",
            "mic": "71",
            "loopback": "72",
        }
        assert payload["native_ec"]["owner_pid"] == 9001
        assert payload["native_ec"]["module_var"] == "1"
        assert ac.inject_target == "cc_autotest_012345abcdef_mic"
        assert ac.capture_source == "cc_autotest_012345abcdef_ec_source"
        assert ac.playback_sink == "cc_autotest_012345abcdef_ec_sink"
        assert ac.needs_routing is False
        alsa_path = ac.child_env["ALSA_CONFIG_PATH"]
        alsa_text = open(alsa_path, encoding="utf-8").read()
        assert payload["alsa_config_path"] == alsa_path
        assert payload["alsa_config_sha256"] == hashlib.sha256(
            alsa_text.encode("utf-8")
        ).hexdigest()
        assert 'capture_node "cc_autotest_012345abcdef_ec_source"' in alsa_text
        assert 'playback_node "cc_autotest_012345abcdef_ec_sink"' in alsa_text

    assert not os.path.exists(path)
    assert not os.path.exists(alsa_path)
    assert ac.cleanup_ok is True
    loads = [
        call for call in modules.calls
        if len(call) > 1 and call[1] == "load-module"
    ]
    assert [call[2] for call in loads] == [
        "module-null-sink",
        "module-null-sink",
        "module-loopback",
    ]
    for call in loads[:2]:
        assert "rate=48000" in call
        assert "channels=1" in call
        assert "channel_map=mono" in call
    assert len(_fake_native_owner) == 1
    assert "buffer.play_delay = 260/1000" in _fake_native_owner[0].config
    assert "audio.channels = 1" in _fake_native_owner[0].config
    assert "audio.position = [ MONO ]" in _fake_native_owner[0].config
    assert "node.dont-reconnect = true" in _fake_native_owner[0].config
    assert _fake_native_owner[0].stop_calls == 1
    assert not any("set-default" in " ".join(call) for call in modules.calls)
    unloads = [
        call[-1] for call in modules.calls
        if len(call) > 1 and call[1] == "unload-module"
    ]
    assert unloads == ["72", "71", "70"]


def test_delay_session_cleans_contract_and_modules_on_exception(monkeypatch):
    modules = _Modules()
    monkeypatch.setattr("tools.autotest.acoustics.subprocess.run", modules)
    ac = DelayAcoustics(prefix="cc_autotest_abcdef012345")

    with pytest.raises(RuntimeError, match="boom"):
        with ac.session():
            path = ac.contract_path
            raise RuntimeError("boom")

    assert path is not None and not os.path.exists(path)
    assert ac.cleanup_ok is True
    assert [
        call[-1] for call in modules.calls
        if len(call) > 1 and call[1] == "unload-module"
    ] == [
        "72", "71", "70",
    ]


def test_hidden_contract_argument_is_scoped_to_delay_child():
    normal = _engine_args("echo", "main", "fast", real_device=False)
    delay = _engine_args(
        "echo",
        "main",
        "fast",
        real_device=False,
        virtual_delay_contract="/tmp/private-route.json",
    )

    assert "--autotest-virtual-delay-contract" not in normal
    index = delay.index("--autotest-virtual-delay-contract")
    assert delay[index + 1] == "/tmp/private-route.json"


@pytest.mark.parametrize(
    "modules",
    (
        _Modules(leaked_module=True),
        _Modules(orphan_stream=True),
        _Modules(orphan_stream_id=True),
        _Modules(orphan_new_stream=True),
        _Modules(default_drift=True),
    ),
)
def test_delay_cleanup_requires_absence_and_default_preservation(
    monkeypatch, modules
):
    monkeypatch.setattr("tools.autotest.acoustics.subprocess.run", modules)
    ac = DelayAcoustics(prefix="cc_autotest_111111111111")

    with ac.session():
        pass

    assert ac.cleanup_ok is False


def test_delay_retains_graph_when_post_child_stream_remains_before_unload(
    monkeypatch, tmp_path
):
    modules = _Modules(residual_child_stream=True)
    monkeypatch.setattr("tools.autotest.acoustics.subprocess.run", modules)
    real_mkstemp = tempfile.mkstemp
    monkeypatch.setattr(
        "tools.autotest.acoustics.tempfile.mkstemp",
        lambda **kwargs: real_mkstemp(dir=tmp_path, **kwargs),
    )
    ac = DelayAcoustics(prefix="cc_autotest_222222222222")

    with ac.session():
        contract_path = ac.contract_path
        alsa_path = ac.child_env["ALSA_CONFIG_PATH"]

    assert ac.cleanup_ok is False
    assert "retained" in ac.cleanup_detail
    assert not any(
        len(call) > 1 and call[1] == "unload-module"
        for call in modules.calls
    )
    assert contract_path is not None and os.path.exists(contract_path)
    assert os.path.exists(alsa_path)
