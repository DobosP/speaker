from __future__ import annotations

import json
import hashlib
import os
import types
from dataclasses import replace

import pytest

from core.virtual_audio import (
    ModuleState,
    NativeEchoCancelContract,
    NativeNodeContract,
    NodeState,
    PactlSnapshot,
    PipeWireClientState,
    PipeWireLinkState,
    PipeWireNodeState,
    PipeWirePortState,
    PipeWireSnapshot,
    PreparedVirtualAudioBinder,
    StreamState,
    VirtualAudioContract,
    VirtualAudioContractError,
    load_virtual_audio_contract,
    probe_pipewire_snapshot,
    render_native_echo_cancel_config,
    render_private_alsa_config,
    validate_virtual_audio_topology,
)


PREFIX = "cc_autotest_012345abcdef"


def _contract(**changes):
    native_config = render_native_echo_cancel_config(
        prefix=PREFIX,
        far_sink=f"{PREFIX}_far",
        mic_sink=f"{PREFIX}_mic",
        latency_ms=260,
    )
    values = {
        "schema": "speaker.autotest.virtual-audio/v2",
        "version": 2,
        "parent_pid": os.getppid(),
        "far_sink": f"{PREFIX}_far",
        "mic_sink": f"{PREFIX}_mic",
        "ec_source": f"{PREFIX}_ec_source",
        "ec_sink": f"{PREFIX}_ec_sink",
        "latency_ms": 260,
        "default_source": "alsa_input.physical",
        "default_sink": "alsa_output.physical",
        "alsa_config_path": f"/tmp/{PREFIX}_asound.conf",
        "alsa_config_sha256": "0" * 64,
        "far_module": "11",
        "mic_module": "12",
        "loopback_module": "13",
        "native_ec": NativeEchoCancelContract(
            owner_pid=os.getppid(),
            client_id="100",
            client_serial="1000",
            config_sha256=hashlib.sha256(
                native_config.encode("utf-8")
            ).hexdigest(),
            module_var="1",
            module_local_id="42",
            capture_node=NativeNodeContract("101", "1001"),
            source_node=NativeNodeContract("102", "1002"),
            sink_node=NativeNodeContract("103", "1003"),
            playback_node=NativeNodeContract("104", "1004"),
        ),
    }
    values.update(changes)
    return VirtualAudioContract(**values)


def _snapshot(contract=None, *, source_outputs=(), sink_inputs=()):
    c = contract or _contract()
    topology_source_outputs = (
        StreamState(
            "61", "21", "loopback.capture", "PipeWire module-loopback",
            c.loopback_module,
        ),
    )
    topology_sink_inputs = (
        StreamState(
            "71", "32", "loopback.playback", "PipeWire module-loopback",
            c.loopback_module,
        ),
    )
    return PactlSnapshot(
        modules=(
            ModuleState(
                c.far_module,
                "module-null-sink",
                f"sink_name={c.far_sink} "
                f"sink_properties=device.description={c.far_sink} "
                "rate=48000 channels=1 channel_map=mono",
            ),
            ModuleState(
                c.mic_module,
                "module-null-sink",
                f"sink_name={c.mic_sink} "
                f"sink_properties=device.description={c.mic_sink} "
                "rate=48000 channels=1 channel_map=mono",
            ),
            ModuleState(
                c.loopback_module,
                "module-loopback",
                f"source={c.far_sink}.monitor sink={c.mic_sink} "
                f"latency_msec={c.latency_ms} source_dont_move=true "
                "sink_dont_move=true",
            ),
        ),
        sources=(
            NodeState("21", f"{c.far_sink}.monitor"),
            NodeState("22", f"{c.mic_sink}.monitor"),
            NodeState("23", c.ec_source),
        ),
        sinks=(
            NodeState("31", c.far_sink),
            NodeState("32", c.mic_sink),
            NodeState("33", c.ec_sink),
        ),
        source_outputs=topology_source_outputs + tuple(source_outputs),
        sink_inputs=topology_sink_inputs + tuple(sink_inputs),
        default_source=c.default_source,
        default_sink=c.default_sink,
    )


def _pw_snapshot(contract=None):
    c = contract or _contract()
    prefix = c.far_sink[:-4]
    group = f"{prefix}_ec"
    nodes = (
        PipeWireNodeState(
            "201", "2001", c.far_sink, "50", "Audio/Sink",
            "", "", "", "", "",
        ),
        PipeWireNodeState(
            "202", "2002", c.mic_sink, "51", "Audio/Sink",
            "", "", "", "", "",
        ),
        PipeWireNodeState(
            c.native_ec.capture_node.object_id,
            c.native_ec.capture_node.serial,
            f"{prefix}_ec_capture",
            c.native_ec.client_id,
            "Stream/Input/Audio",
            group,
            group,
            c.mic_sink,
            prefix,
            str(c.latency_ms),
        ),
        PipeWireNodeState(
            c.native_ec.source_node.object_id,
            c.native_ec.source_node.serial,
            c.ec_source,
            c.native_ec.client_id,
            "Audio/Source",
            group,
            group,
            "",
            prefix,
            str(c.latency_ms),
        ),
        PipeWireNodeState(
            c.native_ec.sink_node.object_id,
            c.native_ec.sink_node.serial,
            c.ec_sink,
            c.native_ec.client_id,
            "Audio/Sink",
            group,
            group,
            "",
            prefix,
            str(c.latency_ms),
        ),
        PipeWireNodeState(
            c.native_ec.playback_node.object_id,
            c.native_ec.playback_node.serial,
            f"{prefix}_ec_playback",
            c.native_ec.client_id,
            "Stream/Output/Audio",
            group,
            group,
            c.far_sink,
            prefix,
            str(c.latency_ms),
        ),
    )
    ports = (
        PipeWirePortState("301", "3001", "101", "0", "in", "MONO"),
        PipeWirePortState("302", "3002", "102", "0", "out", "MONO"),
        PipeWirePortState("303", "3003", "103", "0", "in", "MONO"),
        PipeWirePortState("304", "3004", "104", "0", "out", "MONO"),
        PipeWirePortState("305", "3005", "202", "0", "out", "MONO"),
        PipeWirePortState("306", "3006", "201", "0", "in", "MONO"),
    )
    links = (
        PipeWireLinkState("401", "4001", "202", "305", "101", "301", "paused"),
        PipeWireLinkState("402", "4002", "104", "304", "201", "306", "paused"),
    )
    return PipeWireSnapshot(
        clients=(PipeWireClientState(
            c.native_ec.client_id,
            c.native_ec.client_serial,
            str(c.native_ec.owner_pid),
            str(os.getuid()),
            "pw-cli",
        ),),
        nodes=nodes,
        ports=ports,
        links=links,
    )


def test_exact_virtual_ec_topology_passes():
    detail = validate_virtual_audio_topology(
        _contract(), _snapshot(), _pw_snapshot()
    )

    assert "latency=260ms" in detail
    assert "capture=" in detail and "playback=" in detail


def test_native_ec_renderer_pins_fractional_delay_and_exact_targets():
    rendered = render_native_echo_cancel_config(
        prefix=PREFIX,
        far_sink=f"{PREFIX}_far",
        mic_sink=f"{PREFIX}_mic",
        latency_ms=260,
    )

    assert "buffer.play_delay = 260/1000" in rendered
    assert "audio.rate = 48000" in rendered
    assert "audio.channels = 1" in rendered
    assert "audio.position = [ MONO ]" in rendered
    assert f'target.object = "{PREFIX}_mic"' in rendered
    assert f'target.object = "{PREFIX}_far"' in rendered
    assert rendered.count("node.dont-reconnect = true") == 2
    assert "webrtc.delay_agnostic = true" in rendered


def test_pw_dump_parser_uses_last_complete_snapshot_without_merging():
    pactl = _Pactl()
    first = pactl._pw_dump_text()
    pactl.pipewire = replace(
        pactl.pipewire,
        clients=(replace(pactl.pipewire.clients[0], serial="7777"),),
    )
    second = pactl._pw_dump_text()

    parsed = probe_pipewire_snapshot(
        runner=lambda _argv, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=first + "\n" + second, stderr=""
        )
    )

    assert parsed.clients[0].serial == "7777"


def test_pw_dump_parser_rejects_trailing_partial_snapshot():
    pactl = _Pactl()

    with pytest.raises(VirtualAudioContractError, match="parse"):
        probe_pipewire_snapshot(
            runner=lambda _argv, **_kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=pactl._pw_dump_text() + "\n[{",
                stderr="",
            )
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda s: replace(s, modules=s.modules[:-1]),
        lambda s: replace(
            s,
            modules=s.modules[:2] + (
                replace(s.modules[2], arguments=s.modules[2].arguments.replace(
                    "latency_msec=260", "latency_msec=20"
                )),
            ) + s.modules[3:],
        ),
        lambda s: replace(s, sources=s.sources[:-1]),
        lambda s: replace(s, sinks=s.sinks + (s.sinks[-1],)),
        lambda s: replace(s, default_source="cc_autotest_012345abcdef_ec_source"),
        lambda s: replace(
            s,
            modules=(
                replace(s.modules[0], arguments=s.modules[0].arguments + " extra=true"),
            ) + s.modules[1:],
        ),
        lambda s: replace(
            s,
            modules=(
                replace(
                    s.modules[0],
                    arguments=s.modules[0].arguments.replace("rate=48000 ", ""),
                ),
            ) + s.modules[1:],
        ),
        lambda s: replace(
            s,
            source_outputs=tuple(),
        ),
        lambda s: replace(
            s,
            sink_inputs=tuple(
                replace(stream, owner_module="999")
                if stream.owner_module == "13" else stream
                for stream in s.sink_inputs
            ),
        ),
    ),
)
def test_topology_drift_fails_closed(mutate):
    with pytest.raises(VirtualAudioContractError):
        validate_virtual_audio_topology(
            _contract(), mutate(_snapshot()), _pw_snapshot()
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda s: replace(
            s,
            clients=(replace(s.clients[0], pid="999999"),),
        ),
        lambda s: replace(
            s,
            nodes=s.nodes[:2]
            + (replace(s.nodes[2], serial="9090"),)
            + s.nodes[3:],
        ),
        lambda s: replace(
            s,
            nodes=s.nodes[:2]
            + (replace(s.nodes[2], target_object="physical.source"),)
            + s.nodes[3:],
        ),
        lambda s: replace(
            s,
            nodes=s.nodes[:2]
            + (replace(s.nodes[2], node_group="foreign"),)
            + s.nodes[3:],
        ),
        lambda s: replace(s, links=s.links[:-1]),
        lambda s: replace(
            s,
            nodes=s.nodes
            + (PipeWireNodeState(
                "999", "9999", "physical.source", "77", "Audio/Source",
                "", "", "", "", "",
            ),),
            links=s.links
            + (PipeWireLinkState(
                "999", "9999", "102", "302", "999", "998", "active"
            ),),
        ),
        lambda s: replace(
            s,
            nodes=s.nodes
            + (PipeWireNodeState(
                "998", "9998", f"{PREFIX}_foreign", "77", "Audio/Sink",
                "", "", "", "", "",
            ),),
        ),
    ),
)
def test_native_topology_drift_fails_closed(mutate):
    with pytest.raises(VirtualAudioContractError):
        validate_virtual_audio_topology(
            _contract(), _snapshot(), mutate(_pw_snapshot())
        )


def test_native_config_digest_drift_fails_closed():
    contract = _contract()
    native = replace(contract.native_ec, config_sha256="f" * 64)

    with pytest.raises(VirtualAudioContractError, match="canonical"):
        validate_virtual_audio_topology(
            replace(contract, native_ec=native), _snapshot(contract),
            _pw_snapshot(contract),
        )


def _write_contract(path, contract=None):
    path.write_text(json.dumps((contract or _contract()).to_dict()), encoding="utf-8")
    path.chmod(0o600)
    return path


def _private_contract(tmp_path, **changes):
    path = tmp_path / "private-asound.conf"
    contract = _contract(alsa_config_path=str(path), **changes)
    text = render_private_alsa_config(
        capture_pcm=contract.capture_pcm,
        capture_node=contract.capture_source,
        playback_pcm=contract.playback_pcm,
        playback_node=contract.playback_sink,
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return replace(
        contract,
        alsa_config_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def test_contract_loader_requires_exact_private_parent_bound_file(tmp_path, monkeypatch):
    contract = _private_contract(tmp_path)
    monkeypatch.setenv("ALSA_CONFIG_PATH", contract.alsa_config_path)
    path = _write_contract(tmp_path / "route.json", contract)

    loaded = load_virtual_audio_contract(str(path))

    assert loaded == contract


def test_contract_loader_rejects_permissive_mode_and_parent_mismatch(
    tmp_path, monkeypatch
):
    contract = _private_contract(tmp_path)
    monkeypatch.setenv("ALSA_CONFIG_PATH", contract.alsa_config_path)
    path = _write_contract(tmp_path / "route.json", contract)
    path.chmod(0o640)
    with pytest.raises(VirtualAudioContractError, match="0600"):
        load_virtual_audio_contract(str(path))

    _write_contract(path, replace(contract, parent_pid=os.getppid() + 1))
    with pytest.raises(VirtualAudioContractError, match="parent"):
        load_virtual_audio_contract(str(path))


def test_contract_loader_rejects_symlink_and_unknown_fields(tmp_path, monkeypatch):
    contract = _private_contract(tmp_path)
    monkeypatch.setenv("ALSA_CONFIG_PATH", contract.alsa_config_path)
    real = _write_contract(tmp_path / "real.json", contract)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(VirtualAudioContractError, match="contract"):
        load_virtual_audio_contract(str(link))

    payload = contract.to_dict()
    payload["bypass"] = True
    real.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VirtualAudioContractError, match="unknown"):
        load_virtual_audio_contract(str(real))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("version", True),
        ("parent_pid", float(os.getppid())),
        ("latency_ms", 260.0),
        ("schema", "speaker.autotest.virtual-audio/v3"),
    ),
)
def test_contract_loader_rejects_coercive_or_unknown_schema_values(
    tmp_path, monkeypatch, field, value
):
    path = tmp_path / "route.json"
    contract = _private_contract(tmp_path)
    monkeypatch.setenv("ALSA_CONFIG_PATH", contract.alsa_config_path)
    payload = contract.to_dict()
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(VirtualAudioContractError):
        load_virtual_audio_contract(str(path))


def test_contract_loader_binds_exact_private_alsa_mapping(tmp_path, monkeypatch):
    contract = _private_contract(tmp_path)
    path = _write_contract(tmp_path / "route.json", contract)

    monkeypatch.setenv("ALSA_CONFIG_PATH", str(tmp_path / "other.conf"))
    with pytest.raises(VirtualAudioContractError, match="ALSA_CONFIG_PATH"):
        load_virtual_audio_contract(str(path))

    monkeypatch.setenv("ALSA_CONFIG_PATH", contract.alsa_config_path)
    with open(contract.alsa_config_path, "a", encoding="utf-8") as handle:
        handle.write("# drift\n")
    with pytest.raises(VirtualAudioContractError, match="digest"):
        load_virtual_audio_contract(str(path))


class _Pactl:
    def __init__(self):
        self.contract = _contract()
        self.snapshot = _snapshot()
        self.pipewire = _pw_snapshot()
        self.calls = []

    @staticmethod
    def _stream_text(streams, target_label):
        return "\n\n".join(
            f"{target_label} #{stream.index}\n\t{target_label.split()[0]}: "
            f"{stream.target_index}\n\tOwner Module: {stream.owner_module or 'n/a'}\n"
            "\tProperties:\n"
            f'\t\tnode.name = "{stream.node_name}"\n'
            f'\t\tapplication.name = "{stream.application_name}"'
            for stream in streams
        )

    def _short_modules(self):
        return "\n".join(
            f"{m.module_id}\t{m.name}\t{m.arguments}" for m in self.snapshot.modules
        )

    @staticmethod
    def _short_nodes(nodes):
        return "\n".join(f"{n.index}\t{n.name}\tPipeWire" for n in nodes)

    def _pw_dump_text(self):
        items = []
        for client in self.pipewire.clients:
            items.append({
                "id": int(client.object_id),
                "type": "PipeWire:Interface:Client",
                "info": {"props": {
                    "object.serial": client.serial,
                    "application.process.id": client.pid,
                    "pipewire.sec.uid": client.uid,
                    "application.name": client.application_name,
                }},
            })
        for node in self.pipewire.nodes:
            items.append({
                "id": int(node.object_id),
                "type": "PipeWire:Interface:Node",
                "info": {"props": {
                    "object.serial": node.serial,
                    "node.name": node.name,
                    "client.id": node.client_id,
                    "media.class": node.media_class,
                    "node.group": node.node_group,
                    "node.link-group": node.link_group,
                    "target.object": node.target_object,
                    "speaker.autotest.route": node.route_nonce,
                    "speaker.autotest.latency_ms": node.latency_ms,
                }},
            })
        for port in self.pipewire.ports:
            items.append({
                "id": int(port.object_id),
                "type": "PipeWire:Interface:Port",
                "info": {"props": {
                    "object.serial": port.serial,
                    "node.id": port.node_id,
                    "port.id": port.port_id,
                    "port.direction": port.direction,
                    "audio.channel": port.channel,
                }},
            })
        for link in self.pipewire.links:
            items.append({
                "id": int(link.object_id),
                "type": "PipeWire:Interface:Link",
                "info": {
                    "props": {"object.serial": link.serial},
                    "output-node-id": int(link.output_node_id),
                    "output-port-id": int(link.output_port_id),
                    "input-node-id": int(link.input_node_id),
                    "input-port-id": int(link.input_port_id),
                    "state": link.state,
                },
            })
        return json.dumps(items)

    def __call__(self, argv, **_kwargs):
        self.calls.append(list(argv))
        if list(argv) == ["pw-dump"]:
            return types.SimpleNamespace(
                returncode=0, stdout=self._pw_dump_text(), stderr=""
            )
        args = tuple(argv[1:])
        output = ""
        rc = 0
        if args == ("list", "short", "modules"):
            output = self._short_modules()
        elif args == ("list", "short", "sources"):
            output = self._short_nodes(self.snapshot.sources)
        elif args == ("list", "short", "sinks"):
            output = self._short_nodes(self.snapshot.sinks)
        elif args == ("list", "source-outputs"):
            output = self._stream_text(self.snapshot.source_outputs, "Source Output")
        elif args == ("list", "sink-inputs"):
            output = self._stream_text(self.snapshot.sink_inputs, "Sink Input")
        elif args == ("get-default-source",):
            output = self.snapshot.default_source
        elif args == ("get-default-sink",):
            output = self.snapshot.default_sink
        else:
            rc = 1
        return types.SimpleNamespace(returncode=rc, stdout=output, stderr="bad call")


def test_binder_accepts_only_unique_initially_correct_streams_and_proves_duplex():
    pactl = _Pactl()
    # This pre-existing bridge is part of the baseline and must never move.
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (StreamState(
            "40", "22", "alsa_capture.python", "PipeWire ALSA [python]"
        ),),
    )
    binder = PreparedVirtualAudioBinder.prepare(
        pactl.contract, runner=pactl, poll_interval_sec=0
    )
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (
            StreamState(
                "41", "23", "alsa_capture.python", "PipeWire ALSA [python]"
            ),
        ),
        sink_inputs=pactl.snapshot.sink_inputs + (StreamState(
            "51", "33", "alsa_playback.python", "PipeWire ALSA [python]"
        ),),
    )

    assert binder.bind_capture(timeout_sec=0) == "41"
    assert binder.capture_bound and not binder.playback_bound
    assert binder.verify(require_playback=False)[0]
    assert not binder.fully_bound
    assert binder.bind_playback(timeout_sec=0) == "51"
    assert binder.fully_bound and binder.verify(require_playback=True)[0]

    assert not any(
        len(call) > 1 and call[1].startswith("move-") for call in pactl.calls
    )


def test_binder_rejects_ambiguous_new_streams():
    pactl = _Pactl()
    binder = PreparedVirtualAudioBinder.prepare(pactl.contract, runner=pactl)
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (
            StreamState("41", "22", "alsa_capture.python", "PipeWire ALSA [python]"),
            StreamState("42", "22", "alsa_capture.python", "PipeWire ALSA [python]"),
        ),
    )

    with pytest.raises(VirtualAudioContractError, match="ambiguous"):
        binder.bind_capture(timeout_sec=0)


def test_binder_rejects_wrong_initial_target_without_moving_it():
    pactl = _Pactl()
    binder = PreparedVirtualAudioBinder.prepare(pactl.contract, runner=pactl)
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (StreamState(
            "41", "22", "alsa_capture.python", "PipeWire ALSA [python]"
        ),),
    )

    with pytest.raises(VirtualAudioContractError, match="opened on target"):
        binder.bind_capture(timeout_sec=0)

    assert binder.capture_bound is False
    assert not any(
        len(call) > 1 and call[1].startswith("move-") for call in pactl.calls
    )


def test_binder_detects_post_bind_route_and_topology_drift():
    pactl = _Pactl()
    binder = PreparedVirtualAudioBinder.prepare(pactl.contract, runner=pactl)
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (StreamState(
            "41", "23", "alsa_capture.python", "PipeWire ALSA [python]"
        ),),
    )
    binder.bind_capture(timeout_sec=0)
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=tuple(
            replace(stream, target_index="22")
            if stream.index == "41" else stream
            for stream in pactl.snapshot.source_outputs
        ),
    )
    assert binder.verify(require_playback=False)[0] is False

    pactl.snapshot = replace(pactl.snapshot, modules=pactl.snapshot.modules[:-1])
    ok, detail = binder.verify_topology()
    assert not ok and "module id" in detail


@pytest.mark.parametrize("drift", ("extra", "identity"))
def test_binder_invalidates_exact_delta_authority_on_post_bind_drift(drift):
    pactl = _Pactl()
    binder = PreparedVirtualAudioBinder.prepare(pactl.contract, runner=pactl)
    pactl.snapshot = replace(
        pactl.snapshot,
        source_outputs=pactl.snapshot.source_outputs + (StreamState(
            "41", "23", "alsa_capture.python", "PipeWire ALSA [python]"
        ),),
    )
    binder.bind_capture(timeout_sec=0)

    if drift == "extra":
        outputs = pactl.snapshot.source_outputs + (StreamState(
            "42", "23", "alsa_capture.python.2", "PipeWire ALSA [python]"
        ),)
    else:
        outputs = tuple(
            replace(stream, application_name="PipeWire ALSA [python3]")
            if stream.index == "41" else stream
            for stream in pactl.snapshot.source_outputs
        )
    pactl.snapshot = replace(pactl.snapshot, source_outputs=outputs)

    ok, _detail = binder.verify(require_playback=False)
    assert ok is False
    assert binder.capture_bound is False
