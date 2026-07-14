"""Fail-closed virtual PipeWire route proof for the silent delay harness.

This module is intentionally not configurable through ``config.json``.  The
autonomous harness creates one private manifest for one direct child process;
the child validates the exact run-owned module graph and binds only the new
ALSA-PipeWire streams that appeared after its baseline snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional


class VirtualAudioContractError(RuntimeError):
    """The harness proof was missing, ambiguous, stale, or routed incorrectly."""


_NAME_RE = re.compile(r"^(cc_autotest_[0-9a-f]{12})_far$")
_MODULE_KEYS = frozenset({"far", "mic", "loopback"})
_NATIVE_EC_KEYS = frozenset({
    "owner_pid",
    "client_id",
    "client_serial",
    "config_sha256",
    "module_var",
    "module_local_id",
    "nodes",
})
_NATIVE_NODE_KEYS = frozenset({"capture", "source", "sink", "playback"})
_NATIVE_NODE_RECORD_KEYS = frozenset({"id", "serial"})
_TOP_KEYS = frozenset({
    "schema",
    "version",
    "parent_pid",
    "far_sink",
    "mic_sink",
    "ec_source",
    "ec_sink",
    "latency_ms",
    "default_source",
    "default_sink",
    "alsa_config_path",
    "alsa_config_sha256",
    "modules",
    "native_ec",
})
_SCHEMA = "speaker.autotest.virtual-audio/v2"
_NATIVE_EC_LIBRARY = "aec/libspa-aec-webrtc"


@dataclass(frozen=True)
class NativeNodeContract:
    object_id: str
    serial: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.object_id, "serial": self.serial}


@dataclass(frozen=True)
class NativeEchoCancelContract:
    owner_pid: int
    client_id: str
    client_serial: str
    config_sha256: str
    module_var: str
    module_local_id: str
    capture_node: NativeNodeContract
    source_node: NativeNodeContract
    sink_node: NativeNodeContract
    playback_node: NativeNodeContract

    @property
    def nodes(self) -> Mapping[str, NativeNodeContract]:
        return {
            "capture": self.capture_node,
            "source": self.source_node,
            "sink": self.sink_node,
            "playback": self.playback_node,
        }

    def to_dict(self) -> dict:
        return {
            "owner_pid": self.owner_pid,
            "client_id": self.client_id,
            "client_serial": self.client_serial,
            "config_sha256": self.config_sha256,
            "module_var": self.module_var,
            "module_local_id": self.module_local_id,
            "nodes": {
                name: node.to_dict() for name, node in self.nodes.items()
            },
        }


@dataclass(frozen=True)
class VirtualAudioContract:
    schema: str
    version: int
    parent_pid: int
    far_sink: str
    mic_sink: str
    ec_source: str
    ec_sink: str
    latency_ms: int
    default_source: str
    default_sink: str
    alsa_config_path: str
    alsa_config_sha256: str
    far_module: str
    mic_module: str
    loopback_module: str
    native_ec: NativeEchoCancelContract

    @property
    def capture_source(self) -> str:
        return self.ec_source

    @property
    def playback_sink(self) -> str:
        return self.ec_sink

    @property
    def inject_target(self) -> str:
        return self.mic_sink

    @property
    def capture_pcm(self) -> str:
        return f"{self.far_sink[:-4]}_capture"

    @property
    def playback_pcm(self) -> str:
        return f"{self.far_sink[:-4]}_playback"

    @property
    def modules(self) -> Mapping[str, str]:
        return {
            "far": self.far_module,
            "mic": self.mic_module,
            "loopback": self.loopback_module,
        }

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "version": self.version,
            "parent_pid": self.parent_pid,
            "far_sink": self.far_sink,
            "mic_sink": self.mic_sink,
            "ec_source": self.ec_source,
            "ec_sink": self.ec_sink,
            "latency_ms": self.latency_ms,
            "default_source": self.default_source,
            "default_sink": self.default_sink,
            "alsa_config_path": self.alsa_config_path,
            "alsa_config_sha256": self.alsa_config_sha256,
            "modules": dict(self.modules),
            "native_ec": self.native_ec.to_dict(),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def provenance(self) -> str:
        return f"autotest-virtual-delay:{self.digest}"


@dataclass(frozen=True)
class ModuleState:
    module_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class NodeState:
    index: str
    name: str


@dataclass(frozen=True)
class StreamState:
    index: str
    target_index: str
    node_name: str
    application_name: str = ""
    owner_module: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.node_name, self.application_name


@dataclass(frozen=True)
class PactlSnapshot:
    modules: tuple[ModuleState, ...] = ()
    sources: tuple[NodeState, ...] = ()
    sinks: tuple[NodeState, ...] = ()
    source_outputs: tuple[StreamState, ...] = ()
    sink_inputs: tuple[StreamState, ...] = ()
    default_source: str = ""
    default_sink: str = ""


@dataclass(frozen=True)
class PipeWireClientState:
    object_id: str
    serial: str
    pid: str
    uid: str
    application_name: str


@dataclass(frozen=True)
class PipeWireNodeState:
    object_id: str
    serial: str
    name: str
    client_id: str
    media_class: str
    node_group: str
    link_group: str
    target_object: str
    route_nonce: str
    latency_ms: str


@dataclass(frozen=True)
class PipeWirePortState:
    object_id: str
    serial: str
    node_id: str
    port_id: str
    direction: str
    channel: str


@dataclass(frozen=True)
class PipeWireLinkState:
    object_id: str
    serial: str
    output_node_id: str
    output_port_id: str
    input_node_id: str
    input_port_id: str
    state: str


@dataclass(frozen=True)
class PipeWireSnapshot:
    clients: tuple[PipeWireClientState, ...] = ()
    nodes: tuple[PipeWireNodeState, ...] = ()
    ports: tuple[PipeWirePortState, ...] = ()
    links: tuple[PipeWireLinkState, ...] = ()


Runner = Callable[..., object]


def render_private_alsa_config(
    *,
    capture_pcm: str,
    capture_node: str,
    playback_pcm: str,
    playback_node: str,
) -> str:
    """Return the only ALSA mapping accepted by the private delay contract."""

    return f'''</usr/share/alsa/alsa.conf>

pcm.{capture_pcm} {{
    type pipewire
    capture_node "{capture_node}"
    hint {{
        show on
        description "{capture_pcm}"
    }}
}}

pcm.{playback_pcm} {{
    type pipewire
    playback_node "{playback_node}"
    hint {{
        show on
        description "{playback_pcm}"
    }}
}}
'''


def render_native_echo_cancel_config(
    *,
    prefix: str,
    far_sink: str,
    mic_sink: str,
    latency_ms: int,
) -> str:
    """Render the exact native PipeWire EC module used by the silent gate.

    ``buffer.play_delay`` is a native module-level fraction.  The Pulse
    compatibility ``module-echo-cancel`` does not forward it, which leaves a
    260 ms synthetic air gap effectively uncancelled.  All external streams are
    nonce-named, pinned to the two run-owned null sinks, and forbidden from
    reconnecting to a default device.
    """

    group = f"{prefix}_ec"
    route_props = (
        f'        speaker.autotest.route = "{prefix}"\n'
        f'        speaker.autotest.latency_ms = "{latency_ms}"\n'
    )
    return f'''{{
    remote.name = "pipewire-0"
    node.group = "{group}"
    node.link-group = "{group}"
    audio.rate = 48000
    audio.channels = 1
    audio.position = [ MONO ]
    buffer.play_delay = {latency_ms}/1000
    library.name = "{_NATIVE_EC_LIBRARY}"
    aec.args = {{
        webrtc.noise_suppression = false
        webrtc.gain_control = false
        webrtc.extended_filter = true
        webrtc.delay_agnostic = true
    }}
    capture.props = {{
        node.name = "{prefix}_ec_capture"
        target.object = "{mic_sink}"
        stream.capture.sink = true
        node.dont-reconnect = true
{route_props}    }}
    source.props = {{
        node.name = "{prefix}_ec_source"
        node.description = "{prefix}_ec_source"
{route_props}    }}
    sink.props = {{
        node.name = "{prefix}_ec_sink"
        node.description = "{prefix}_ec_sink"
{route_props}    }}
    playback.props = {{
        node.name = "{prefix}_ec_playback"
        target.object = "{far_sink}"
        node.dont-reconnect = true
{route_props}    }}
}}
'''


def _run_text(runner: Runner, *args: str) -> str:
    try:
        result = runner(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:  # noqa: BLE001 - proof probes fail closed
        raise VirtualAudioContractError(f"pactl {' '.join(args)} failed: {exc}") from exc
    if int(getattr(result, "returncode", 1)) != 0:
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise VirtualAudioContractError(
            f"pactl {' '.join(args)} failed" + (f": {detail}" if detail else "")
        )
    return str(getattr(result, "stdout", "") or "")


def _run_command_text(runner: Runner, argv: list[str]) -> str:
    try:
        result = runner(argv, capture_output=True, text=True, timeout=3)
    except Exception as exc:  # noqa: BLE001 - proof probes fail closed
        raise VirtualAudioContractError(f"{' '.join(argv)} failed: {exc}") from exc
    if int(getattr(result, "returncode", 1)) != 0:
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise VirtualAudioContractError(
            f"{' '.join(argv)} failed" + (f": {detail}" if detail else "")
        )
    return str(getattr(result, "stdout", "") or "")


def _prop_text(props: Mapping[str, object], key: str) -> str:
    value = props.get(key, "")
    return "" if value is None else str(value)


def _object_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (2**31 - 1, text)


def probe_pipewire_snapshot(*, runner: Runner = subprocess.run) -> PipeWireSnapshot:
    """Read native PipeWire clients/nodes/ports/links from one JSON dump."""

    text = _run_command_text(runner, ["pw-dump"])
    try:
        decoder = json.JSONDecoder()
        documents = []
        cursor = 0
        while cursor < len(text):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text):
                break
            document, cursor = decoder.raw_decode(text, cursor)
            documents.append(document)
            if len(documents) > 16:
                raise ValueError("too many concatenated snapshots")
    except (TypeError, ValueError) as exc:
        raise VirtualAudioContractError(f"cannot parse pw-dump JSON: {exc}") from exc
    if not documents or any(not isinstance(item, list) for item in documents):
        raise VirtualAudioContractError("pw-dump did not contain complete list snapshots")
    # pw-dump can restart and append a fresh complete array when the graph
    # changes during enumeration.  Never merge generations; the surrounding
    # stable-route sampler still requires the final complete snapshot from two
    # separate invocations to agree.
    raw = documents[-1]

    clients: list[PipeWireClientState] = []
    nodes: list[PipeWireNodeState] = []
    ports: list[PipeWirePortState] = []
    links: list[PipeWireLinkState] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id", ""))
        kind = str(item.get("type", ""))
        info = item.get("info") if isinstance(item.get("info"), dict) else {}
        props = info.get("props") if isinstance(info.get("props"), dict) else {}
        serial = _prop_text(props, "object.serial")
        if kind.startswith("PipeWire:Interface:Client"):
            clients.append(PipeWireClientState(
                object_id=object_id,
                serial=serial,
                pid=(
                    _prop_text(props, "application.process.id")
                    or _prop_text(props, "pipewire.sec.pid")
                ),
                uid=_prop_text(props, "pipewire.sec.uid"),
                application_name=_prop_text(props, "application.name"),
            ))
        elif kind.startswith("PipeWire:Interface:Node"):
            nodes.append(PipeWireNodeState(
                object_id=object_id,
                serial=serial,
                name=_prop_text(props, "node.name"),
                client_id=_prop_text(props, "client.id"),
                media_class=_prop_text(props, "media.class"),
                node_group=_prop_text(props, "node.group"),
                link_group=_prop_text(props, "node.link-group"),
                target_object=_prop_text(props, "target.object"),
                route_nonce=_prop_text(props, "speaker.autotest.route"),
                latency_ms=_prop_text(props, "speaker.autotest.latency_ms"),
            ))
        elif kind.startswith("PipeWire:Interface:Port"):
            ports.append(PipeWirePortState(
                object_id=object_id,
                serial=serial,
                node_id=_prop_text(props, "node.id"),
                port_id=_prop_text(props, "port.id"),
                direction=_prop_text(props, "port.direction"),
                channel=(
                    _prop_text(props, "audio.channel")
                    or _prop_text(props, "port.name")
                ),
            ))
        elif kind.startswith("PipeWire:Interface:Link"):
            links.append(PipeWireLinkState(
                object_id=object_id,
                serial=serial,
                output_node_id=str(info.get("output-node-id", "")),
                output_port_id=str(info.get("output-port-id", "")),
                input_node_id=str(info.get("input-node-id", "")),
                input_port_id=str(info.get("input-port-id", "")),
                state=str(info.get("state", "")),
            ))
    key = lambda value: _object_sort_key(value.object_id)
    return PipeWireSnapshot(
        clients=tuple(sorted(clients, key=key)),
        nodes=tuple(sorted(nodes, key=key)),
        ports=tuple(sorted(ports, key=key)),
        links=tuple(sorted(links, key=key)),
    )


def _parse_modules(text: str) -> tuple[ModuleState, ...]:
    out = []
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 2 and parts[0].strip().isdigit():
            out.append(ModuleState(
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip() if len(parts) > 2 else "",
            ))
    return tuple(out)


def _parse_nodes(text: str) -> tuple[NodeState, ...]:
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            out.append(NodeState(parts[0].strip(), parts[1].strip()))
    return tuple(out)


def _parse_streams(text: str) -> tuple[StreamState, ...]:
    out = []
    for block in re.split(r"\n\s*\n", text):
        index = re.search(r"#(\d+)", block)
        target = re.search(r"(?:Sink|Source):\s*(\d+)", block)
        if index is None or target is None:
            continue
        node = re.search(r'node\.name\s*=\s*"([^"]+)"', block)
        app = re.search(r'application\.name\s*=\s*"([^"]+)"', block)
        owner = re.search(r"Owner Module:\s*(\S+)", block)
        out.append(StreamState(
            index.group(1),
            target.group(1),
            node.group(1) if node else "",
            app.group(1) if app else "",
            owner.group(1) if owner else "",
        ))
    return tuple(out)


def probe_pactl_snapshot(*, runner: Runner = subprocess.run) -> PactlSnapshot:
    """Read one complete pactl proof snapshot; any missing query is fatal."""

    return PactlSnapshot(
        modules=_parse_modules(_run_text(runner, "list", "short", "modules")),
        sources=_parse_nodes(_run_text(runner, "list", "short", "sources")),
        sinks=_parse_nodes(_run_text(runner, "list", "short", "sinks")),
        source_outputs=_parse_streams(_run_text(runner, "list", "source-outputs")),
        sink_inputs=_parse_streams(_run_text(runner, "list", "sink-inputs")),
        default_source=_run_text(runner, "get-default-source").strip(),
        default_sink=_run_text(runner, "get-default-sink").strip(),
    )


def _probe_stable_pactl_snapshot(
    *, runner: Runner = subprocess.run, attempts: int = 4
) -> PactlSnapshot:
    """Require two consecutive complete inventories to agree.

    ``pactl`` cannot return modules, nodes, streams, and defaults atomically.
    Consecutive equality prevents an authority decision from joining pieces of
    different route graphs. One bounded retry window lets a new stream settle;
    continuous churn fails closed.
    """

    previous = probe_pactl_snapshot(runner=runner)
    for _ in range(max(1, int(attempts))):
        current = probe_pactl_snapshot(runner=runner)
        if current == previous:
            return current
        previous = current
    raise VirtualAudioContractError("PipeWire route changed while proof was sampled")


def _pipewire_projection(
    contract: VirtualAudioContract,
    snapshot: PipeWireSnapshot,
) -> tuple[tuple[object, ...], ...]:
    """Return only graph objects capable of changing this contract's route."""

    prefix = contract.far_sink[:-4]
    native_ids = {
        node.object_id for node in contract.native_ec.nodes.values()
    }
    relevant_ids = {
        node.object_id
        for node in snapshot.nodes
        if node.object_id in native_ids or node.name.startswith(prefix)
    }
    relevant_links = tuple(
        link for link in snapshot.links
        if (
            link.output_node_id in relevant_ids
            or link.input_node_id in relevant_ids
        )
    )
    adjacent_ids = relevant_ids | {
        endpoint
        for link in relevant_links
        for endpoint in (link.output_node_id, link.input_node_id)
    }
    nodes = tuple(node for node in snapshot.nodes if node.object_id in adjacent_ids)
    client_ids = {node.client_id for node in nodes}
    clients = tuple(
        client for client in snapshot.clients
        if client.object_id in client_ids
        and client.application_name != "pw-dump"
    )
    ports = tuple(port for port in snapshot.ports if port.node_id in adjacent_ids)
    return clients, nodes, ports, relevant_links


def _probe_stable_virtual_route(
    contract: VirtualAudioContract,
    *,
    runner: Runner = subprocess.run,
    attempts: int = 4,
) -> tuple[PactlSnapshot, PipeWireSnapshot]:
    """Require two consecutive Pulse and relevant native inventories to agree."""

    previous_pactl = probe_pactl_snapshot(runner=runner)
    previous_pipewire = probe_pipewire_snapshot(runner=runner)
    previous_projection = _pipewire_projection(contract, previous_pipewire)
    for _ in range(max(1, int(attempts))):
        current_pactl = probe_pactl_snapshot(runner=runner)
        current_pipewire = probe_pipewire_snapshot(runner=runner)
        current_projection = _pipewire_projection(contract, current_pipewire)
        if (
            current_pactl == previous_pactl
            and current_projection == previous_projection
        ):
            return current_pactl, current_pipewire
        previous_pactl = current_pactl
        previous_pipewire = current_pipewire
        previous_projection = current_projection
    raise VirtualAudioContractError(
        "PipeWire virtual route changed while proof was sampled"
    )


def _module_args(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise VirtualAudioContractError(f"malformed module arguments: {exc}") from exc
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise VirtualAudioContractError(
                f"malformed module argument token {token!r}"
            )
        # PipeWire-Pulse drops the argv quoting around module-echo-cancel's
        # space-separated aec_args in `list short modules`. Reassemble only
        # dotted WebRTC continuations after the declared aec_args key; the
        # final exact-map comparison still rejects extras, duplicates, or drift.
        if key.startswith("webrtc.") and "aec_args" in out:
            out["aec_args"] += f" {token}"
            continue
        if key in out:
            raise VirtualAudioContractError(f"duplicate module argument {key!r}")
        out[key] = value
    return out


def _one_by_id(snapshot: PactlSnapshot, module_id: str) -> ModuleState:
    found = [module for module in snapshot.modules if module.module_id == module_id]
    if len(found) != 1:
        raise VirtualAudioContractError(
            f"module id {module_id} appears {len(found)} times"
        )
    return found[0]


def _require_module(
    snapshot: PactlSnapshot,
    module_id: str,
    expected_name: str,
    expected_args: Mapping[str, str],
) -> ModuleState:
    module = _one_by_id(snapshot, module_id)
    if module.name != expected_name:
        raise VirtualAudioContractError(
            f"module {module_id} is {module.name!r}, expected {expected_name!r}"
        )
    args = _module_args(module.arguments)
    if args != dict(expected_args):
        raise VirtualAudioContractError(
            f"module {module_id} arguments differ from the exact contract"
        )
    return module


def _require_unique_node(nodes: tuple[NodeState, ...], name: str, kind: str) -> NodeState:
    found = [node for node in nodes if node.name == name]
    if len(found) != 1:
        raise VirtualAudioContractError(
            f"{kind} node {name!r} appears {len(found)} times"
        )
    return found[0]


def _require_owned_stream(
    streams: tuple[StreamState, ...],
    *,
    module_id: str,
    target_index: str,
    kind: str,
) -> StreamState:
    found = [stream for stream in streams if stream.owner_module == module_id]
    if len(found) != 1:
        raise VirtualAudioContractError(
            f"module {module_id} owns {len(found)} {kind} streams"
        )
    if found[0].target_index != target_index:
        raise VirtualAudioContractError(
            f"module {module_id} {kind} targets {found[0].target_index}, "
            f"expected {target_index}"
        )
    return found[0]


def _one_pipewire_node(
    snapshot: PipeWireSnapshot,
    *,
    name: str,
) -> PipeWireNodeState:
    found = [node for node in snapshot.nodes if node.name == name]
    if len(found) != 1:
        raise VirtualAudioContractError(
            f"native PipeWire node {name!r} appears {len(found)} times"
        )
    return found[0]


def _native_node_names(prefix: str) -> dict[str, str]:
    return {
        "capture": f"{prefix}_ec_capture",
        "source": f"{prefix}_ec_source",
        "sink": f"{prefix}_ec_sink",
        "playback": f"{prefix}_ec_playback",
    }


def attest_native_echo_cancel(
    *,
    prefix: str,
    far_sink: str,
    mic_sink: str,
    latency_ms: int,
    owner_pid: int,
    config_sha256: str,
    module_var: str,
    module_local_id: str,
    snapshot: PipeWireSnapshot,
) -> NativeEchoCancelContract:
    """Bind the parent's retained pw-cli owner to exact remote node IDs."""

    names = _native_node_names(prefix)
    observed = {
        key: _one_pipewire_node(snapshot, name=name)
        for key, name in names.items()
    }
    owner_client_ids = {node.client_id for node in observed.values()}
    if len(owner_client_ids) != 1:
        raise VirtualAudioContractError(
            "native EC nodes do not share one owning client"
        )
    owner_client_id = next(iter(owner_client_ids))
    clients = [
        client for client in snapshot.clients
        if (
            client.object_id == owner_client_id
            and client.pid == str(owner_pid)
            and client.application_name == "pw-cli"
        )
    ]
    if len(clients) != 1:
        raise VirtualAudioContractError(
            "native EC node-owning pw-cli client identity is ambiguous"
        )
    client = clients[0]
    native = NativeEchoCancelContract(
        owner_pid=int(owner_pid),
        client_id=client.object_id,
        client_serial=client.serial,
        config_sha256=config_sha256,
        module_var=module_var,
        module_local_id=module_local_id,
        capture_node=NativeNodeContract(
            observed["capture"].object_id, observed["capture"].serial
        ),
        source_node=NativeNodeContract(
            observed["source"].object_id, observed["source"].serial
        ),
        sink_node=NativeNodeContract(
            observed["sink"].object_id, observed["sink"].serial
        ),
        playback_node=NativeNodeContract(
            observed["playback"].object_id, observed["playback"].serial
        ),
    )
    _validate_native_echo_cancel(
        native,
        prefix=prefix,
        far_sink=far_sink,
        mic_sink=mic_sink,
        latency_ms=latency_ms,
        snapshot=snapshot,
    )
    return native


def _port_channels(
    snapshot: PipeWireSnapshot,
    *,
    node_id: str,
    direction: str,
) -> tuple[PipeWirePortState, ...]:
    return tuple(
        port for port in snapshot.ports
        if port.node_id == node_id and port.direction == direction
    )


def _normalized_channel(port: PipeWirePortState) -> str:
    return port.channel.rsplit("_", 1)[-1].upper()


def _validate_native_echo_cancel(
    native: NativeEchoCancelContract,
    *,
    prefix: str,
    far_sink: str,
    mic_sink: str,
    latency_ms: int,
    snapshot: PipeWireSnapshot,
) -> None:
    if (
        type(native.owner_pid) is not int
        or native.owner_pid <= 0
        or not native.client_id.isdigit()
        or not native.client_serial.isdigit()
        or not native.module_var.isdigit()
        or not native.module_local_id.isdigit()
        or re.fullmatch(r"[0-9a-f]{64}", native.config_sha256) is None
    ):
        raise VirtualAudioContractError("native EC ownership fields are invalid")
    expected_config = render_native_echo_cancel_config(
        prefix=prefix,
        far_sink=far_sink,
        mic_sink=mic_sink,
        latency_ms=latency_ms,
    )
    if hashlib.sha256(expected_config.encode("utf-8")).hexdigest() != native.config_sha256:
        raise VirtualAudioContractError("native EC config digest is not canonical")

    clients = [
        client for client in snapshot.clients
        if client.object_id == native.client_id
    ]
    if len(clients) != 1:
        raise VirtualAudioContractError(
            f"native EC client {native.client_id} appears {len(clients)} times"
        )
    client = clients[0]
    if (
        client.serial != native.client_serial
        or client.pid != str(native.owner_pid)
        or client.uid != str(os.getuid())
        or client.application_name != "pw-cli"
    ):
        raise VirtualAudioContractError("native EC owner client identity drifted")

    names = _native_node_names(prefix)
    expected_classes = {
        "capture": "Stream/Input/Audio",
        "source": "Audio/Source",
        "sink": "Audio/Sink",
        "playback": "Stream/Output/Audio",
    }
    expected_targets = {
        "capture": mic_sink,
        "source": "",
        "sink": "",
        "playback": far_sink,
    }
    group = f"{prefix}_ec"
    observed: dict[str, PipeWireNodeState] = {}
    for key, name in names.items():
        node = _one_pipewire_node(snapshot, name=name)
        expected = native.nodes[key]
        if (
            node.object_id != expected.object_id
            or node.serial != expected.serial
            or node.client_id != native.client_id
            or node.media_class != expected_classes[key]
            or node.node_group != group
            or node.link_group != group
            or node.target_object != expected_targets[key]
            or node.route_nonce != prefix
            or node.latency_ms != str(latency_ms)
        ):
            raise VirtualAudioContractError(
                f"native EC {key} node differs from the exact contract"
            )
        observed[key] = node

    expected_prefix_names = {
        far_sink,
        mic_sink,
        *names.values(),
    }
    actual_prefix_names = {
        node.name for node in snapshot.nodes if node.name.startswith(prefix)
    }
    if actual_prefix_names != expected_prefix_names:
        raise VirtualAudioContractError(
            "native EC generated node namespace has missing or foreign nodes"
        )
    far_node = _one_pipewire_node(snapshot, name=far_sink)
    mic_node = _one_pipewire_node(snapshot, name=mic_sink)

    required_ports: dict[str, tuple[PipeWirePortState, ...]] = {
        "capture": _port_channels(
            snapshot, node_id=observed["capture"].object_id, direction="in"
        ),
        "source": _port_channels(
            snapshot, node_id=observed["source"].object_id, direction="out"
        ),
        "sink": _port_channels(
            snapshot, node_id=observed["sink"].object_id, direction="in"
        ),
        "playback": _port_channels(
            snapshot, node_id=observed["playback"].object_id, direction="out"
        ),
    }
    for key, ports in required_ports.items():
        if (
            len(ports) != 1
            or _normalized_channel(ports[0]) != "MONO"
            or any(not port.object_id.isdigit() or not port.serial.isdigit() for port in ports)
        ):
            raise VirtualAudioContractError(
                f"native EC {key} lacks one exact MONO "
                f"{ports[0].direction if ports else ''} port"
            )

    allowed_states = {"paused", "active", "running"}
    capture_port_ids = {port.object_id for port in required_ports["capture"]}
    playback_port_ids = {port.object_id for port in required_ports["playback"]}
    capture_links = [
        link for link in snapshot.links
        if link.input_port_id in capture_port_ids
    ]
    playback_links = [
        link for link in snapshot.links
        if link.output_port_id in playback_port_ids
    ]
    if (
        len(capture_links) != 1
        or any(
            link.output_node_id != mic_node.object_id
            or link.input_node_id != observed["capture"].object_id
            or link.state not in allowed_states
            for link in capture_links
        )
    ):
        raise VirtualAudioContractError(
            "native EC capture links are not exact mic-monitor links"
        )
    if (
        len(playback_links) != 1
        or any(
            link.output_node_id != observed["playback"].object_id
            or link.input_node_id != far_node.object_id
            or link.state not in allowed_states
            for link in playback_links
        )
    ):
        raise VirtualAudioContractError(
            "native EC playback links are not exact far-sink links"
        )

    ports_by_id = {port.object_id: port for port in snapshot.ports}
    mic_link_port = ports_by_id.get(capture_links[0].output_port_id)
    far_link_port = ports_by_id.get(playback_links[0].input_port_id)
    if (
        mic_link_port is None
        or mic_link_port.node_id != mic_node.object_id
        or mic_link_port.direction != "out"
        or _normalized_channel(mic_link_port) != "MONO"
        or far_link_port is None
        or far_link_port.node_id != far_node.object_id
        or far_link_port.direction != "in"
        or _normalized_channel(far_link_port) != "MONO"
    ):
        raise VirtualAudioContractError(
            "native EC external links are not exact MONO null-sink ports"
        )

    node_by_id = {node.object_id: node for node in snapshot.nodes}
    native_ids = {node.object_id for node in observed.values()}
    for link in snapshot.links:
        if not (
            link.output_node_id in native_ids or link.input_node_id in native_ids
        ):
            continue
        pair = (link.output_node_id, link.input_node_id)
        if pair in {
            (mic_node.object_id, observed["capture"].object_id),
            (observed["playback"].object_id, far_node.object_id),
        }:
            continue
        output = node_by_id.get(link.output_node_id)
        input_node = node_by_id.get(link.input_node_id)
        if (
            output is not None
            and input_node is not None
            and output.object_id == observed["source"].object_id
            and input_node.name.startswith("alsa_capture")
        ):
            continue
        if (
            output is not None
            and input_node is not None
            and output.name.startswith("alsa_playback")
            and input_node.object_id == observed["sink"].object_id
        ):
            continue
        raise VirtualAudioContractError(
            "native EC node has a foreign or physical PipeWire link"
        )


def validate_virtual_audio_topology(
    contract: VirtualAudioContract,
    snapshot: PactlSnapshot,
    pipewire_snapshot: PipeWireSnapshot,
) -> str:
    """Validate the exact Pulse/native graph and return compact proof detail."""

    _validate_contract_values(contract)
    _require_module(
        snapshot,
        contract.far_module,
        "module-null-sink",
        {
            "sink_name": contract.far_sink,
            "sink_properties": f"device.description={contract.far_sink}",
            "rate": "48000",
            "channels": "1",
            "channel_map": "mono",
        },
    )
    _require_module(
        snapshot,
        contract.mic_module,
        "module-null-sink",
        {
            "sink_name": contract.mic_sink,
            "sink_properties": f"device.description={contract.mic_sink}",
            "rate": "48000",
            "channels": "1",
            "channel_map": "mono",
        },
    )
    _require_module(
        snapshot,
        contract.loopback_module,
        "module-loopback",
        {
            "source": f"{contract.far_sink}.monitor",
            "sink": contract.mic_sink,
            "latency_msec": str(contract.latency_ms),
            "source_dont_move": "true",
            "sink_dont_move": "true",
        },
    )
    if (
        snapshot.default_source != contract.default_source
        or snapshot.default_sink != contract.default_sink
    ):
        raise VirtualAudioContractError("system default source/sink changed during run")
    generated_nodes = {
        contract.far_sink,
        contract.mic_sink,
        contract.ec_source,
        contract.ec_sink,
    }
    if contract.default_source in generated_nodes or contract.default_sink in generated_nodes:
        raise VirtualAudioContractError("virtual nodes must never become system defaults")
    owned_ids = set(contract.modules.values())
    prefix = contract.far_sink[:-4]
    for module in snapshot.modules:
        if module.module_id not in owned_ids and prefix in module.arguments:
            raise VirtualAudioContractError(
                f"foreign module {module.module_id} touches the generated namespace"
            )

    far_sink = _require_unique_node(snapshot.sinks, contract.far_sink, "sink")
    mic_sink = _require_unique_node(snapshot.sinks, contract.mic_sink, "sink")
    _require_unique_node(snapshot.sinks, contract.ec_sink, "sink")
    far_monitor = _require_unique_node(
        snapshot.sources, f"{contract.far_sink}.monitor", "source"
    )
    mic_monitor = _require_unique_node(
        snapshot.sources, f"{contract.mic_sink}.monitor", "source"
    )
    _require_unique_node(snapshot.sources, contract.ec_source, "source")

    # Module arguments are declarations. The live backing streams are the
    # physical-device boundary, so prove both modules still own exactly their
    # declared source and sink masters.
    _require_owned_stream(
        snapshot.source_outputs,
        module_id=contract.loopback_module,
        target_index=far_monitor.index,
        kind="source-output",
    )
    _require_owned_stream(
        snapshot.sink_inputs,
        module_id=contract.loopback_module,
        target_index=mic_sink.index,
        kind="sink-input",
    )
    del mic_monitor, far_sink
    _validate_native_echo_cancel(
        contract.native_ec,
        prefix=prefix,
        far_sink=contract.far_sink,
        mic_sink=contract.mic_sink,
        latency_ms=contract.latency_ms,
        snapshot=pipewire_snapshot,
    )
    return (
        f"digest={contract.digest} capture={contract.ec_source} "
        f"playback={contract.ec_sink} latency={contract.latency_ms}ms "
        f"native_ec_pid={contract.native_ec.owner_pid}"
    )


def _validate_contract_values(contract: VirtualAudioContract) -> None:
    if type(contract.schema) is not str or contract.schema != _SCHEMA:
        raise VirtualAudioContractError(
            f"unsupported contract schema {contract.schema!r}"
        )
    if type(contract.version) is not int or contract.version != 2:
        raise VirtualAudioContractError(
            f"unsupported contract version {contract.version!r}"
        )
    if (
        type(contract.parent_pid) is not int
        or contract.parent_pid <= 0
        or contract.parent_pid != os.getppid()
    ):
        raise VirtualAudioContractError(
            f"contract parent {contract.parent_pid} does not match {os.getppid()}"
        )
    string_fields = (
        contract.far_sink,
        contract.mic_sink,
        contract.ec_source,
        contract.ec_sink,
        contract.default_source,
        contract.default_sink,
        contract.alsa_config_path,
        contract.alsa_config_sha256,
        contract.native_ec.client_id,
        contract.native_ec.client_serial,
        contract.native_ec.config_sha256,
        contract.native_ec.module_var,
        contract.native_ec.module_local_id,
        *contract.modules.values(),
        *(
            value
            for node in contract.native_ec.nodes.values()
            for value in (node.object_id, node.serial)
        ),
    )
    if any(type(value) is not str or not value for value in string_fields):
        raise VirtualAudioContractError("contract string fields must be non-empty strings")
    match = _NAME_RE.fullmatch(contract.far_sink)
    if match is None:
        raise VirtualAudioContractError("far sink is not a generated autotest name")
    prefix = match.group(1)
    expected_names = {
        "mic_sink": f"{prefix}_mic",
        "ec_source": f"{prefix}_ec_source",
        "ec_sink": f"{prefix}_ec_sink",
    }
    for field_name, expected in expected_names.items():
        if getattr(contract, field_name) != expected:
            raise VirtualAudioContractError(
                f"{field_name} does not share the generated run prefix"
            )
    if type(contract.latency_ms) is not int or not 1 <= contract.latency_ms <= 2000:
        raise VirtualAudioContractError("latency_ms must be in 1..2000")
    if not os.path.isabs(contract.alsa_config_path):
        raise VirtualAudioContractError("ALSA config path must be absolute")
    if re.fullmatch(r"[0-9a-f]{64}", contract.alsa_config_sha256) is None:
        raise VirtualAudioContractError("ALSA config digest must be full SHA-256")
    if (
        type(contract.native_ec.owner_pid) is not int
        or contract.native_ec.owner_pid <= 0
        or re.fullmatch(r"[0-9a-f]{64}", contract.native_ec.config_sha256) is None
    ):
        raise VirtualAudioContractError("native EC owner/config fields are invalid")
    native_ids = (
        contract.native_ec.client_id,
        contract.native_ec.client_serial,
        contract.native_ec.module_var,
        contract.native_ec.module_local_id,
        *(
            value
            for node in contract.native_ec.nodes.values()
            for value in (node.object_id, node.serial)
        ),
    )
    if any(not value.isdigit() or int(value) <= 0 for value in native_ids):
        raise VirtualAudioContractError(
            "native EC IDs and serials must be positive integer strings"
        )
    node_ids = tuple(
        node.object_id for node in contract.native_ec.nodes.values()
    )
    node_serials = tuple(
        node.serial for node in contract.native_ec.nodes.values()
    )
    if len(set(node_ids)) != 4 or len(set(node_serials)) != 4:
        raise VirtualAudioContractError("native EC node IDs/serials must be unique")
    module_ids = tuple(contract.modules.values())
    if (
        any(not str(value).isdigit() or int(value) <= 0 for value in module_ids)
        or len(set(module_ids)) != len(module_ids)
    ):
        raise VirtualAudioContractError("module IDs must be unique positive integers")


def _validate_private_alsa_mapping(contract: VirtualAudioContract) -> None:
    if os.environ.get("ALSA_CONFIG_PATH") != contract.alsa_config_path:
        raise VirtualAudioContractError(
            "ALSA_CONFIG_PATH does not match the private contract"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(contract.alsa_config_path, flags)
    except OSError as exc:
        raise VirtualAudioContractError(f"cannot open private ALSA config: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise VirtualAudioContractError(
                "ALSA config must be one regular file, not a symlink/hardlink"
            )
        if info.st_uid != os.getuid():
            raise VirtualAudioContractError("ALSA config owner does not match current uid")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise VirtualAudioContractError("ALSA config mode must be exactly 0600")
        if info.st_size <= 0 or info.st_size > 16_384:
            raise VirtualAudioContractError("ALSA config size is outside 1..16384 bytes")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            payload = handle.read(16_385)
    finally:
        if fd >= 0:
            os.close(fd)
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != contract.alsa_config_sha256:
        raise VirtualAudioContractError("ALSA config digest does not match contract")
    expected = render_private_alsa_config(
        capture_pcm=contract.capture_pcm,
        capture_node=contract.capture_source,
        playback_pcm=contract.playback_pcm,
        playback_node=contract.playback_sink,
    ).encode("utf-8")
    if payload != expected:
        raise VirtualAudioContractError(
            "ALSA config does not exactly map the private capture/playback nodes"
        )


def load_virtual_audio_contract(path: str) -> VirtualAudioContract:
    """Load one regular uid-owned exact-0600 manifest bound to this parent."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VirtualAudioContractError(f"cannot open private contract: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise VirtualAudioContractError(
                "contract must be one regular file, not a symlink/hardlink"
            )
        if info.st_uid != os.getuid():
            raise VirtualAudioContractError("contract owner does not match current uid")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise VirtualAudioContractError("contract mode must be exactly 0600")
        if info.st_size <= 0 or info.st_size > 16_384:
            raise VirtualAudioContractError("contract size is outside 1..16384 bytes")
        with os.fdopen(fd, encoding="utf-8") as handle:
            fd = -1

            def _unique_object(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise VirtualAudioContractError(
                            f"duplicate JSON field {key!r}"
                        )
                    obj[key] = value
                return obj

            raw = json.load(handle, object_pairs_hook=_unique_object)
    except VirtualAudioContractError:
        raise
    except (OSError, ValueError) as exc:
        raise VirtualAudioContractError(f"cannot read contract JSON: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
        raise VirtualAudioContractError("contract has missing or unknown fields")
    modules = raw.get("modules")
    if not isinstance(modules, dict) or set(modules) != _MODULE_KEYS:
        raise VirtualAudioContractError("contract modules have missing or unknown fields")
    native_raw = raw.get("native_ec")
    if not isinstance(native_raw, dict) or set(native_raw) != _NATIVE_EC_KEYS:
        raise VirtualAudioContractError(
            "contract native_ec has missing or unknown fields"
        )
    native_nodes = native_raw.get("nodes")
    if not isinstance(native_nodes, dict) or set(native_nodes) != _NATIVE_NODE_KEYS:
        raise VirtualAudioContractError(
            "contract native EC nodes have missing or unknown fields"
        )
    if any(
        not isinstance(node, dict) or set(node) != _NATIVE_NODE_RECORD_KEYS
        for node in native_nodes.values()
    ):
        raise VirtualAudioContractError("contract native EC node records are invalid")
    if (
        type(raw["version"]) is not int
        or type(raw["parent_pid"]) is not int
        or type(raw["latency_ms"]) is not int
        or type(native_raw["owner_pid"]) is not int
        or any(
            type(raw[key]) is not str
            for key in (
                "schema", "far_sink", "mic_sink", "ec_source", "ec_sink",
                "default_source", "default_sink", "alsa_config_path",
                "alsa_config_sha256",
            )
        )
        or any(type(value) is not str for value in modules.values())
        or any(
            type(native_raw[key]) is not str
            for key in (
                "client_id", "client_serial", "config_sha256",
                "module_var", "module_local_id",
            )
        )
        or any(
            type(value) is not str
            for node in native_nodes.values()
            for value in node.values()
        )
    ):
        raise VirtualAudioContractError("contract fields have non-exact JSON types")
    native = NativeEchoCancelContract(
        owner_pid=native_raw["owner_pid"],
        client_id=native_raw["client_id"],
        client_serial=native_raw["client_serial"],
        config_sha256=native_raw["config_sha256"],
        module_var=native_raw["module_var"],
        module_local_id=native_raw["module_local_id"],
        capture_node=NativeNodeContract(
            native_nodes["capture"]["id"], native_nodes["capture"]["serial"]
        ),
        source_node=NativeNodeContract(
            native_nodes["source"]["id"], native_nodes["source"]["serial"]
        ),
        sink_node=NativeNodeContract(
            native_nodes["sink"]["id"], native_nodes["sink"]["serial"]
        ),
        playback_node=NativeNodeContract(
            native_nodes["playback"]["id"], native_nodes["playback"]["serial"]
        ),
    )
    contract = VirtualAudioContract(
        schema=raw["schema"],
        version=raw["version"],
        parent_pid=raw["parent_pid"],
        far_sink=raw["far_sink"],
        mic_sink=raw["mic_sink"],
        ec_source=raw["ec_source"],
        ec_sink=raw["ec_sink"],
        latency_ms=raw["latency_ms"],
        default_source=raw["default_source"],
        default_sink=raw["default_sink"],
        alsa_config_path=raw["alsa_config_path"],
        alsa_config_sha256=raw["alsa_config_sha256"],
        far_module=modules["far"],
        mic_module=modules["mic"],
        loopback_module=modules["loopback"],
        native_ec=native,
    )
    _validate_contract_values(contract)
    _validate_private_alsa_mapping(contract)
    return contract


def _is_engine_stream(stream: StreamState, kind: str) -> bool:
    prefix = "alsa_capture" if kind == "capture" else "alsa_playback"
    return bool(
        (stream.node_name == prefix or stream.node_name.startswith(prefix + "."))
        and re.fullmatch(r"PipeWire ALSA \[python[^\]]*\]", stream.application_name)
    )


@dataclass
class PreparedVirtualAudioBinder:
    """Two-phase exact-ID binder prepared before PortAudio opens."""

    contract: VirtualAudioContract
    runner: Runner = field(default=subprocess.run, repr=False)
    poll_interval_sec: float = 0.05
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    topology_detail: str = ""
    _baseline_capture: frozenset[str] = field(default_factory=frozenset, repr=False)
    _baseline_playback: frozenset[str] = field(default_factory=frozenset, repr=False)
    _capture_id: Optional[str] = field(default=None, repr=False)
    _playback_id: Optional[str] = field(default=None, repr=False)
    _capture_fingerprint: Optional[tuple[str, str]] = field(default=None, repr=False)
    _playback_fingerprint: Optional[tuple[str, str]] = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def prepare(
        cls,
        contract: VirtualAudioContract,
        *,
        runner: Runner = subprocess.run,
        poll_interval_sec: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "PreparedVirtualAudioBinder":
        snapshot, pipewire = _probe_stable_virtual_route(
            contract, runner=runner
        )
        detail = validate_virtual_audio_topology(contract, snapshot, pipewire)
        return cls(
            contract=contract,
            runner=runner,
            poll_interval_sec=max(0.0, float(poll_interval_sec)),
            clock=clock,
            sleep=sleep,
            topology_detail=detail,
            _baseline_capture=frozenset(
                stream.index for stream in snapshot.source_outputs
            ),
            _baseline_playback=frozenset(
                stream.index for stream in snapshot.sink_inputs
            ),
        )

    @property
    def capture_bound(self) -> bool:
        with self._lock:
            return self._capture_id is not None

    @property
    def playback_bound(self) -> bool:
        with self._lock:
            return self._playback_id is not None

    @property
    def fully_bound(self) -> bool:
        with self._lock:
            return self._capture_id is not None and self._playback_id is not None

    def verify_topology(self) -> tuple[bool, str]:
        with self._lock:
            try:
                snapshot, pipewire = _probe_stable_virtual_route(
                    self.contract, runner=self.runner
                )
                detail = validate_virtual_audio_topology(
                    self.contract, snapshot, pipewire
                )
            except VirtualAudioContractError as exc:
                self._invalidate_unlocked()
                return False, str(exc)
            self.topology_detail = detail
            return True, detail

    def _node_index(
        self, snapshot: PactlSnapshot, kind: str, name: str
    ) -> str:
        nodes = snapshot.sources if kind == "capture" else snapshot.sinks
        return _require_unique_node(nodes, name, kind).index

    def _streams(self, snapshot: PactlSnapshot, kind: str) -> tuple[StreamState, ...]:
        return snapshot.source_outputs if kind == "capture" else snapshot.sink_inputs

    def _bound_id(self, kind: str) -> Optional[str]:
        return self._capture_id if kind == "capture" else self._playback_id

    def _set_bound_id(self, kind: str, value: Optional[str]) -> None:
        if kind == "capture":
            self._capture_id = value
            self._capture_fingerprint = None
        else:
            self._playback_id = value
            self._playback_fingerprint = None

    def _fingerprint(self, kind: str) -> Optional[tuple[str, str]]:
        return (
            self._capture_fingerprint
            if kind == "capture"
            else self._playback_fingerprint
        )

    def _set_fingerprint(self, kind: str, stream: StreamState) -> None:
        if kind == "capture":
            self._capture_fingerprint = stream.identity
        else:
            self._playback_fingerprint = stream.identity

    def _candidates(
        self,
        snapshot: PactlSnapshot,
        kind: str,
        baseline: frozenset[str],
    ) -> list[StreamState]:
        return [
            stream for stream in self._streams(snapshot, kind)
            if stream.index not in baseline and _is_engine_stream(stream, kind)
        ]

    def _bind(self, kind: str, *, timeout_sec: float) -> str:
        if kind not in {"capture", "playback"}:
            raise ValueError(kind)
        target_name = (
            self.contract.capture_source
            if kind == "capture"
            else self.contract.playback_sink
        )
        baseline = (
            self._baseline_capture if kind == "capture" else self._baseline_playback
        )
        deadline = self.clock() + max(0.0, float(timeout_sec))
        while True:
            snapshot, pipewire = _probe_stable_virtual_route(
                self.contract, runner=self.runner
            )
            validate_virtual_audio_topology(
                self.contract, snapshot, pipewire
            )
            current_id = self._bound_id(kind)
            candidates = self._candidates(snapshot, kind, baseline)
            if current_id is not None:
                expected = self._node_index(snapshot, kind, target_name)
                fingerprint = self._fingerprint(kind)
                if (
                    len(candidates) == 1
                    and candidates[0].index == current_id
                    and candidates[0].identity == fingerprint
                    and candidates[0].target_index == expected
                ):
                    return current_id
                self._set_bound_id(kind, None)
                raise VirtualAudioContractError(
                    f"bound {kind} stream changed or became ambiguous"
                )
            if len(candidates) > 1:
                raise VirtualAudioContractError(
                    f"ambiguous new {kind} streams: {[item.index for item in candidates]}"
                )
            if len(candidates) == 1:
                stream = candidates[0]
                expected = self._node_index(snapshot, kind, target_name)
                if stream.target_index != expected:
                    raise VirtualAudioContractError(
                        f"{kind} stream {stream.index} opened on target "
                        f"{stream.target_index}, expected {target_name}"
                    )
                # The private ALSA PCM must select the EC node at creation.
                # Moving a wrong-target stream after open could already have
                # activated a physical device, so this authority never moves.
                self._set_bound_id(kind, stream.index)
                self._set_fingerprint(kind, stream)
                return stream.index
            if self.clock() >= deadline:
                raise VirtualAudioContractError(f"no unique new {kind} stream appeared")
            self.sleep(self.poll_interval_sec)

    def bind_capture(self, *, timeout_sec: float = 5.0) -> str:
        with self._lock:
            return self._bind("capture", timeout_sec=timeout_sec)

    def bind_playback(self, *, timeout_sec: float = 5.0) -> str:
        with self._lock:
            return self._bind("playback", timeout_sec=timeout_sec)

    def verify(self, *, require_playback: bool) -> tuple[bool, str]:
        with self._lock:
            try:
                snapshot, pipewire = _probe_stable_virtual_route(
                    self.contract, runner=self.runner
                )
                detail = validate_virtual_audio_topology(
                    self.contract, snapshot, pipewire
                )
                checks = (("capture", self._capture_id),)
                if require_playback or self._playback_id is not None:
                    checks += (("playback", self._playback_id),)
                for kind, stream_id in checks:
                    if stream_id is None:
                        raise VirtualAudioContractError(f"{kind} stream is not bound")
                    target_name = (
                        self.contract.capture_source
                        if kind == "capture"
                        else self.contract.playback_sink
                    )
                    expected = self._node_index(snapshot, kind, target_name)
                    baseline = (
                        self._baseline_capture
                        if kind == "capture"
                        else self._baseline_playback
                    )
                    candidates = self._candidates(snapshot, kind, baseline)
                    if (
                        len(candidates) != 1
                        or candidates[0].index != stream_id
                        or candidates[0].identity != self._fingerprint(kind)
                        or candidates[0].target_index != expected
                    ):
                        raise VirtualAudioContractError(
                            f"bound {kind} stream drifted from {target_name}"
                        )
            except VirtualAudioContractError as exc:
                self._invalidate_unlocked()
                return False, str(exc)
            return True, detail

    def _invalidate_unlocked(self, kind: Optional[str] = None) -> None:
        if kind in {None, "capture"}:
            self._capture_id = None
            self._capture_fingerprint = None
        if kind in {None, "playback"}:
            self._playback_id = None
            self._playback_fingerprint = None

    def invalidate(self, kind: Optional[str] = None) -> None:
        with self._lock:
            if kind not in {None, "capture", "playback"}:
                raise ValueError(kind)
            self._invalidate_unlocked(kind)
