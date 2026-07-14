"""Acoustic-path backends for the voice tier -- how the assistant's TTS reaches
the engine's mic, and where injected 'user' utterances go.

* :class:`CableAcoustics` -- a single PipeWire null sink (digital loopback,
  ~tens-of-ms delay). Silent, fast, parallel-safe. Default.
* :class:`DelayAcoustics` -- two 48 kHz mono null sinks bridged by a pinned
  ``module-loopback`` at the declared delay, wrapped by a retained, run-owned
  native PipeWire echo-cancel module carrying the same ``buffer.play_delay``.
  Exact private ALSA PCMs open the EC source/sink directly; user clips enter the
  raw mic sink as near-field input. Silent, default-preserving, and hardware-free.
* :class:`SpeakerAcoustics` -- no virtual devices: the engine runs on the real
  default speaker + mic, and user clips play out the real speaker. TRUE
  over-the-air (real ~260 ms acoustic delay, real speaker/room coloring) -- the
  genuine open-speaker condition. Makes audible sound and records the real mic.

Each is a context manager that loads/unloads its PipeWire modules and exposes:
  ``needs_routing``   -- whether engine streams must be moved (False for speaker)
  ``inject_target``   -- pactl sink to paplay user clips into (None = default)
  ``route(pid)``      -- move the engine's streams onto this rig
  ``capture_ready(pid)`` -- engine capture is attached to the right source
  ``uses_real_device``-- engine should run on default devices (speaker) vs the
                         ``pipewire`` ALSA bridge (cable/delay)
"""
from __future__ import annotations

import contextlib
from collections import deque
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from typing import Iterator, Optional

from core.virtual_audio import (
    NativeEchoCancelContract,
    VirtualAudioContractError,
    attest_native_echo_cancel,
    probe_pipewire_snapshot,
    render_native_echo_cancel_config,
    render_private_alsa_config,
)

from . import audio


class CableAcoustics:
    """Two null sinks: the engine PLAYS to a dead ``play`` sink (discarded) and
    CAPTURES a separate ``cap`` sink where clips are injected. The assistant's
    TTS therefore never reaches the mic -> NO echo -> clean, reproducible STT +
    round-trip (digital injection = a perfect near-field user). It does NOT test
    self-interrupt or barge-in -- both need the echo/talk-over relationship; use
    ``delay`` (silent) or ``speaker`` (real over-the-air)."""

    needs_routing = True
    uses_real_device = False
    has_echo = False     # playback -> dead sink, so STT only (no self-interrupt/barge)
    inject_gain = 55         # the engine captures the monitor ~2.7x hot; keep it
                             # below clipping so the echo/quiet floor doesn't
                             # ratchet up + start dropping clips on long runs
    inject_lead_in_ms = 0    # virtual cable: instant, no warm-up needed

    def __init__(self, prefix: str = "cc_autotest"):
        self._play = f"{prefix}_play"
        self._cap = f"{prefix}_cap"
        self._mods: list[str] = []
        self._cleanup_permitted = True

    def retain_for_live_child(self) -> None:
        self._cleanup_permitted = False

    def release_after_child_exit(self) -> None:
        """Permit teardown only after the runtime proves child quiescence."""

        self._cleanup_permitted = True

    @property
    def inject_target(self) -> Optional[str]:
        return self._cap

    @property
    def capture_source(self) -> str:
        return f"{self._cap}.monitor"

    @contextlib.contextmanager
    def session(self) -> Iterator["CableAcoustics"]:
        self._cleanup_permitted = True
        def load(name: str) -> None:
            mid = subprocess.run(
                ["pactl", "load-module", "module-null-sink", f"sink_name={name}",
                 f"sink_properties=device.description={name}"],
                capture_output=True, text=True, check=True).stdout.strip()
            if not mid.isdigit():
                raise RuntimeError(f"load-module failed: {name} -> {mid!r}")
            self._mods.append(mid)
            subprocess.run(["pactl", "set-sink-volume", name, "100%"], capture_output=True)

        try:
            load(self._play)
            load(self._cap)
            yield self
        finally:
            if self._cleanup_permitted:
                for mid in reversed(self._mods):
                    subprocess.run(["pactl", "unload-module", mid], capture_output=True)
                self._mods.clear()

    def route(self, pid: int) -> None:
        audio.route_streams(pid, self._play, f"{self._cap}.monitor")

    def capture_ready(self, pid: int) -> bool:
        return audio.capture_on(pid, f"{self._cap}.monitor")


class _NativeEchoCancelOwner:
    """Retained interactive pw-cli process that owns one native EC module."""

    _MODULE_RE = re.compile(r"(?:^|\s)(\d+)\s*=\s*@module:(\d+)(?:\s|$)")

    def __init__(self, config: str):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.module_var = ""
        self.module_local_id = ""
        self._reader: Optional[threading.Thread] = None
        self._module_ready = threading.Event()
        self._reader_done = threading.Event()
        self._tail: deque[str] = deque(maxlen=160)

    @property
    def pid(self) -> int:
        return int(self.process.pid) if self.process is not None else 0

    @property
    def detail(self) -> str:
        return "\n".join(self._tail)[-4000:]

    def _read_output(self) -> None:
        try:
            stream = self.process.stdout if self.process is not None else None
            if stream is None:
                return
            for raw in stream:
                line = raw.rstrip("\r\n")
                self._tail.append(line)
                match = self._MODULE_RE.search(line)
                if match is not None and not self.module_var:
                    self.module_var, self.module_local_id = match.groups()
                    self._module_ready.set()
        finally:
            self._reader_done.set()

    def start(self, *, timeout_sec: float = 5.0) -> None:
        self.process = subprocess.Popen(
            ["pw-cli"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name="autotest-native-ec-reader",
            daemon=True,
        )
        self._reader.start()
        stdin = self.process.stdin
        if stdin is None:
            self.stop()
            raise RuntimeError("pw-cli native EC command channel is unavailable")
        stdin.write(
            "load-module libpipewire-module-echo-cancel "
            f"{self.config.replace(chr(10), ' ')}\n"
        )
        stdin.flush()
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while not self._module_ready.wait(timeout=0.05):
            if self.process.poll() is not None or time.monotonic() >= deadline:
                detail = self.detail
                self.stop()
                raise RuntimeError(
                    "native PipeWire EC module did not publish ownership"
                    + (f": {detail}" if detail else "")
                )

    def stop(self, *, timeout_sec: float = 3.0) -> tuple[bool, str]:
        process = self.process
        if process is None:
            return True, "native EC owner was not started"
        if process.poll() is None:
            stdin = process.stdin
            if stdin is not None:
                with contextlib.suppress(Exception):
                    if self.module_var:
                        stdin.write(f"unload-module {self.module_var}\n")
                    stdin.write("quit\n")
                    stdin.flush()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=timeout_sec)
        if process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=timeout_sec)
        if process.poll() is None:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=timeout_sec)
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        if self._reader is not None:
            self._reader.join(timeout=timeout_sec)
        quiesced = process.poll() is not None and (
            self._reader is None or not self._reader.is_alive()
        )
        return quiesced, (
            "native EC owner exited"
            if quiesced
            else "native EC owner/reader did not quiesce"
        )


class DelayAcoustics:
    # The child process owns exact before/after stream-ID binding through the
    # private contract. The legacy broad mover is unsafe for this mode.
    needs_routing = False
    uses_real_device = False
    has_echo = True
    # The verified native route can present up to ~4x the paplay waveform at the
    # capture source even when every node is mono. Pulse volume is perceptual:
    # 63% maps to ~0.25 linear amplitude (0.63 ** 3), so a 0.12-RMS / 0.80-peak
    # synthesized clip arrives near 0.12 / <=0.80. This leaves the production
    # boost-only AGC and GTCRN in their clean domain.
    inject_gain = 63
    # The EC suppresses a simultaneous near-end command by roughly another
    # order of magnitude. Unity left only isolated pre-gain blocks above the
    # calibrated floor. Pulse's perceptual 130% is ~2.20x linear, still far from
    # the command window's measured capture headroom, and models a close near-end
    # talker without heating the ordinary WER clips.
    barge_inject_gain = 130
    inject_lead_in_ms = 0    # virtual sinks: no resume/warm-up to cover
    # A newly opened paplay stream otherwise loses/corrupts its first command
    # word at the EC/VAD boundary. Same-stream silence warms that exact path;
    # causal latency starts after this pad, at the declared speech onset.
    barge_lead_in_ms = 300

    def __init__(self, latency_ms: int = 260, prefix: Optional[str] = None):
        self.latency_ms = latency_ms
        base = prefix or f"cc_autotest_{secrets.token_hex(6)}"
        self._far = f"{base}_far"
        self._mic = f"{base}_mic"
        self._ec_source = f"{base}_ec_source"
        self._ec_sink = f"{base}_ec_sink"
        self._capture_pcm = f"{base}_capture"
        self._playback_pcm = f"{base}_playback"
        self._mods: list[str] = []
        self._contract_path: Optional[str] = None
        self._alsa_config_path: Optional[str] = None
        self.cleanup_ok = False
        self.cleanup_detail = "not run"
        self._cleanup_permitted = True
        self._default_source = ""
        self._default_sink = ""
        self._pre_child_stream_ids: dict[str, frozenset[str]] = {}
        self._native_owner: Optional[_NativeEchoCancelOwner] = None
        self._native_contract: Optional[NativeEchoCancelContract] = None

    def retain_for_live_child(self) -> None:
        self._cleanup_permitted = False
        self.cleanup_ok = False
        self.cleanup_detail = "retained: engine child may still be live"

    def release_after_child_exit(self) -> None:
        """Permit teardown only after the runtime proves child quiescence."""

        self._cleanup_permitted = True

    def _start_native_echo_cancel(self, config: str) -> _NativeEchoCancelOwner:
        owner = _NativeEchoCancelOwner(config)
        owner.start()
        return owner

    def _attest_native_owner(
        self,
        owner: _NativeEchoCancelOwner,
        *,
        config_sha256: str,
        timeout_sec: float = 5.0,
    ) -> NativeEchoCancelContract:
        prefix = self._far[:-4]
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        previous: Optional[NativeEchoCancelContract] = None
        last_error = "native EC graph did not settle"
        while time.monotonic() <= deadline:
            if owner.process is None or owner.process.poll() is not None:
                raise RuntimeError(
                    "native EC owner exited before graph attestation"
                    + (f": {owner.detail}" if owner.detail else "")
                )
            try:
                snapshot = probe_pipewire_snapshot()
                current = attest_native_echo_cancel(
                    prefix=prefix,
                    far_sink=self._far,
                    mic_sink=self._mic,
                    latency_ms=int(self.latency_ms),
                    owner_pid=owner.pid,
                    config_sha256=config_sha256,
                    module_var=owner.module_var,
                    module_local_id=owner.module_local_id,
                    snapshot=snapshot,
                )
                if current == previous:
                    return current
                previous = current
            except VirtualAudioContractError as exc:
                previous = None
                last_error = str(exc)
            time.sleep(0.05)
        raise RuntimeError(f"native EC graph attestation failed: {last_error}")

    def _native_graph_absent(self, *, timeout_sec: float = 3.0) -> bool:
        native = self._native_contract
        names = {
            f"{self._far[:-4]}_ec_capture",
            self._ec_source,
            self._ec_sink,
            f"{self._far[:-4]}_ec_playback",
        }
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        consecutive = 0
        while time.monotonic() <= deadline:
            try:
                snapshot = probe_pipewire_snapshot()
            except VirtualAudioContractError:
                consecutive = 0
            else:
                nodes_absent = not any(node.name in names for node in snapshot.nodes)
                client_absent = native is None or not any(
                    client.object_id == native.client_id
                    and client.serial == native.client_serial
                    for client in snapshot.clients
                )
                if nodes_absent and client_absent:
                    consecutive += 1
                    if consecutive >= 2:
                        return True
                else:
                    consecutive = 0
            time.sleep(0.05)
        return False

    def _snapshot_owned_stream_ids(
        self, owner_ids: tuple[str, ...]
    ) -> tuple[bool, dict[str, frozenset[str]], str]:
        inventories: dict[str, str] = {}
        for key, kind in (
            ("source_outputs", "source-outputs"),
            ("sink_inputs", "sink-inputs"),
        ):
            result = subprocess.run(
                ["pactl", "list", kind], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False, {}, f"could not snapshot {kind} before unload"
            inventories[key] = result.stdout or ""
        captured: dict[str, frozenset[str]] = {}
        for kind, text in inventories.items():
            pairs = []
            for block in re.split(r"\n\s*\n", text):
                index = re.search(r"#(\d+)", block)
                owner = re.search(r"Owner Module:\s*(\d+)", block)
                if index is not None and owner is not None and owner.group(1) in owner_ids:
                    pairs.append((index.group(1), owner.group(1)))
            for owner_id in owner_ids:
                if sum(owner == owner_id for _index, owner in pairs) != 1:
                    return (
                        False,
                        {},
                        f"module {owner_id} lacks one exact {kind} before unload",
                    )
            captured[kind] = frozenset(index for index, _owner in pairs)
        return True, captured, "owned backing stream IDs snapshotted"

    def _snapshot_all_stream_ids(
        self,
    ) -> tuple[bool, dict[str, frozenset[str]], str]:
        captured: dict[str, frozenset[str]] = {}
        for key, kind in (
            ("source_outputs", "source-outputs"),
            ("sink_inputs", "sink-inputs"),
        ):
            result = subprocess.run(
                ["pactl", "list", kind], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False, {}, f"could not snapshot pre-child {kind} IDs"
            captured[key] = frozenset(
                re.findall(r"#(\d+)", result.stdout or "")
            )
        return True, captured, "pre-child stream IDs snapshotted"

    def _verify_cleanup(
        self,
        module_ids: tuple[str, ...],
        owned_stream_ids: dict[str, frozenset[str]],
        pre_child_stream_ids: dict[str, frozenset[str]],
    ) -> tuple[bool, str]:
        probes: dict[str, str] = {}
        commands = {
            "modules": ["pactl", "list", "short", "modules"],
            "sources": ["pactl", "list", "short", "sources"],
            "sinks": ["pactl", "list", "short", "sinks"],
            "source_outputs": ["pactl", "list", "source-outputs"],
            "sink_inputs": ["pactl", "list", "sink-inputs"],
            "default_source": ["pactl", "get-default-source"],
            "default_sink": ["pactl", "get-default-sink"],
        }
        for key, command in commands.items():
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"cleanup probe failed: {key}"
            probes[key] = result.stdout or ""
        pipewire = subprocess.run(
            ["pw-dump"], capture_output=True, text=True
        )
        if pipewire.returncode != 0:
            return False, "cleanup probe failed: pw-dump"
        probes["pipewire"] = pipewire.stdout or ""
        if probes["default_source"].strip() != self._default_source:
            return False, "default source changed"
        if probes["default_sink"].strip() != self._default_sink:
            return False, "default sink changed"
        remaining_module_ids = {
            line.split("\t", 1)[0].strip()
            for line in probes["modules"].splitlines()
            if line.split("\t", 1)[0].strip().isdigit()
        }
        leaked_ids = sorted(set(module_ids) & remaining_module_ids)
        if leaked_ids:
            return False, f"generated modules remain: {leaked_ids}"
        stream_owner_ids = set(
            re.findall(
                r"Owner Module:\s*(\d+)",
                probes["source_outputs"] + "\n" + probes["sink_inputs"],
            )
        )
        leaked_stream_owners = sorted(set(module_ids) & stream_owner_ids)
        if leaked_stream_owners:
            return False, (
                "streams owned by generated modules remain: "
                f"{leaked_stream_owners}"
            )
        for kind, old_ids in owned_stream_ids.items():
            remaining_ids = set(re.findall(r"#(\d+)", probes[kind]))
            leaked_stream_ids = sorted(old_ids & remaining_ids)
            if leaked_stream_ids:
                return False, (
                    f"generated {kind} IDs remain: {leaked_stream_ids}"
                )
            unexpected_ids = sorted(
                remaining_ids - set(pre_child_stream_ids.get(kind, frozenset()))
            )
            if unexpected_ids:
                return False, (
                    f"post-child {kind} IDs remain outside baseline: "
                    f"{unexpected_ids}"
                )
        prefix = self._far[:-4]
        inventories = "\n".join(
            probes[key]
            for key in (
                "modules", "sources", "sinks", "source_outputs", "sink_inputs",
                "pipewire",
            )
        )
        if prefix in inventories:
            return False, "generated nodes or streams remain"
        return True, "modules/nodes/streams absent; defaults preserved"

    @property
    def inject_target(self) -> Optional[str]:
        return self._mic        # near-field: into the mic sink, no extra delay

    @property
    def capture_source(self) -> str:
        return self._ec_source

    @property
    def playback_sink(self) -> str:
        return self._ec_sink

    @property
    def contract_path(self) -> Optional[str]:
        return self._contract_path

    @property
    def child_env(self) -> dict[str, str]:
        return (
            {"ALSA_CONFIG_PATH": self._alsa_config_path}
            if self._alsa_config_path
            else {}
        )

    @contextlib.contextmanager
    def session(self) -> Iterator["DelayAcoustics"]:
        self.cleanup_ok = False
        self.cleanup_detail = "not run"
        self._cleanup_permitted = True
        self._pre_child_stream_ids = {}
        self._native_owner = None
        self._native_contract = None
        def load(args: list[str]) -> str:
            mid = subprocess.run(["pactl", "load-module", *args],
                                 capture_output=True, text=True, check=True).stdout.strip()
            if not mid.isdigit():
                raise RuntimeError(f"load-module failed: {args} -> {mid!r}")
            self._mods.append(mid)
            return mid

        try:
            default_source = subprocess.run(
                ["pactl", "get-default-source"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            default_sink = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if not default_source or not default_sink:
                raise RuntimeError("could not snapshot PipeWire defaults")
            self._default_source = default_source
            self._default_sink = default_sink
            far_module = load([
                "module-null-sink",
                f"sink_name={self._far}",
                f"sink_properties=device.description={self._far}",
                "rate=48000",
                "channels=1",
                "channel_map=mono",
            ])
            mic_module = load([
                "module-null-sink",
                f"sink_name={self._mic}",
                f"sink_properties=device.description={self._mic}",
                "rate=48000",
                "channels=1",
                "channel_map=mono",
            ])
            subprocess.run(["pactl", "set-sink-volume", self._far, "100%"], capture_output=True)
            subprocess.run(["pactl", "set-sink-volume", self._mic, "100%"], capture_output=True)
            # the air gap: far.monitor --(delayed)--> mic, pinned so the engine's
            # own moves don't drag the bridge endpoints around.
            loopback_module = load([
                "module-loopback",
                f"source={self._far}.monitor",
                f"sink={self._mic}",
                f"latency_msec={self.latency_ms}",
                "source_dont_move=true",
                "sink_dont_move=true",
            ])
            prefix = self._far[:-4]
            native_config = render_native_echo_cancel_config(
                prefix=prefix,
                far_sink=self._far,
                mic_sink=self._mic,
                latency_ms=int(self.latency_ms),
            )
            native_digest = hashlib.sha256(
                native_config.encode("utf-8")
            ).hexdigest()
            self._native_owner = self._start_native_echo_cancel(native_config)
            self._native_contract = self._attest_native_owner(
                self._native_owner,
                config_sha256=native_digest,
            )
            subprocess.run(
                ["pactl", "set-sink-volume", self._ec_sink, "100%"],
                capture_output=True,
            )
            alsa_fd, alsa_path = tempfile.mkstemp(
                prefix=f"{self._far[:-4]}_", suffix="_asound.conf"
            )
            self._alsa_config_path = alsa_path
            try:
                os.fchmod(alsa_fd, 0o600)
                alsa_text = render_private_alsa_config(
                    capture_pcm=self._capture_pcm,
                    capture_node=self._ec_source,
                    playback_pcm=self._playback_pcm,
                    playback_node=self._ec_sink,
                )
                with os.fdopen(alsa_fd, "w", encoding="utf-8") as handle:
                    alsa_fd = -1
                    handle.write(alsa_text)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if alsa_fd >= 0:
                    os.close(alsa_fd)
            alsa_digest = hashlib.sha256(alsa_text.encode("utf-8")).hexdigest()
            fd, path = tempfile.mkstemp(
                prefix=f"{self._far[:-4]}_", suffix="_route.json"
            )
            self._contract_path = path
            try:
                os.fchmod(fd, 0o600)
                payload = {
                    "schema": "speaker.autotest.virtual-audio/v2",
                    "version": 2,
                    "parent_pid": os.getpid(),
                    "far_sink": self._far,
                    "mic_sink": self._mic,
                    "ec_source": self._ec_source,
                    "ec_sink": self._ec_sink,
                    "latency_ms": int(self.latency_ms),
                    "default_source": default_source,
                    "default_sink": default_sink,
                    "alsa_config_path": alsa_path,
                    "alsa_config_sha256": alsa_digest,
                    "modules": {
                        "far": far_module,
                        "mic": mic_module,
                        "loopback": loopback_module,
                    },
                    "native_ec": self._native_contract.to_dict(),
                }
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    fd = -1
                    json.dump(payload, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            baseline_ok, baseline_ids, baseline_detail = self._snapshot_all_stream_ids()
            if not baseline_ok:
                raise RuntimeError(baseline_detail)
            self._pre_child_stream_ids = baseline_ids
            yield self
        finally:
            if not self._cleanup_permitted:
                self.cleanup_ok = False
                self.cleanup_detail = "retained: engine child may still be live"
            else:
                cleanup_ok = True
                module_ids = tuple(self._mods)
                current_ok, current_ids, current_detail = self._snapshot_all_stream_ids()
                baseline_ids = self._pre_child_stream_ids or current_ids
                if len(self._mods) >= 3:
                    stream_snapshot_ok, owned_stream_ids, stream_detail = (
                        self._snapshot_owned_stream_ids((self._mods[-1],))
                    )
                else:
                    stream_snapshot_ok = True
                    owned_stream_ids = {}
                    stream_detail = "partial setup had no child backing graph"
                confirm_ok, confirm_ids, confirm_detail = self._snapshot_all_stream_ids()
                if not confirm_ok or confirm_ids != current_ids:
                    current_ok = False
                    current_detail = (
                        confirm_detail
                        if not confirm_ok
                        else "stream inventory changed during pre-unload proof"
                    )
                unexpected_before_unload = {
                    kind: sorted(ids - baseline_ids.get(kind, frozenset()))
                    for kind, ids in confirm_ids.items()
                    if ids - baseline_ids.get(kind, frozenset())
                }
                pre_unload_ok = (
                    current_ok
                    and stream_snapshot_ok
                    and not unexpected_before_unload
                )
                if not pre_unload_ok:
                    self.cleanup_ok = False
                    if unexpected_before_unload:
                        self.cleanup_detail = (
                            "retained: post-child streams remain before unload: "
                            f"{unexpected_before_unload}"
                        )
                    elif not current_ok:
                        self.cleanup_detail = f"retained: {current_detail}"
                    else:
                        self.cleanup_detail = f"retained: {stream_detail}"
                else:
                    native_ok = True
                    native_detail = "native EC owner was not started"
                    if self._native_owner is not None:
                        native_ok, native_detail = self._native_owner.stop()
                    native_absent = native_ok and self._native_graph_absent()
                    if not native_ok or not native_absent:
                        self.cleanup_ok = False
                        self.cleanup_detail = (
                            "retained: " + native_detail
                            if not native_ok
                            else "retained: native EC nodes/client remain"
                        )
                    else:
                        self._native_owner = None
                        if self._alsa_config_path:
                            try:
                                os.unlink(self._alsa_config_path)
                            except OSError:
                                cleanup_ok = False
                            else:
                                self._alsa_config_path = None
                        if self._contract_path:
                            try:
                                os.unlink(self._contract_path)
                            except OSError:
                                cleanup_ok = False
                            else:
                                self._contract_path = None
                        for mid in reversed(self._mods):
                            result = subprocess.run(
                                ["pactl", "unload-module", mid], capture_output=True
                            )
                            cleanup_ok = cleanup_ok and result.returncode == 0
                        self._mods.clear()
                        proof_ok, detail = self._verify_cleanup(
                            module_ids,
                            owned_stream_ids,
                            baseline_ids,
                        )
                        self.cleanup_ok = cleanup_ok and proof_ok
                        self.cleanup_detail = detail

    def route(self, pid: int) -> None:  # exact binding is owned by the child
        pass

    def capture_ready(self, pid: int) -> bool:
        return audio.capture_on(pid, self._ec_source)


class SpeakerAcoustics:
    """Real over-the-air: the engine speaks out the default sink + captures the
    real mic. ``inject_sink`` controls where the 'user' clips play:

    * ``None`` -> the default sink (assistant and user share one speaker; both
      far-field). Simplest.
    * a sink name (e.g. the laptop's ``alsa_output...analog-stereo``) -> the
      user clips come from a DIFFERENT speaker than the assistant. With the
      assistant on an external speaker (e.g. a JBL set as default) and the user
      clips on the laptop speaker next to the mic, this gives real near/far
      separation -- the faithful open-speaker scenario.
    """

    needs_routing = False
    uses_real_device = True
    has_echo = True
    capture_source = ""
    # real over-the-air: pad each injected clip with lead-in silence so the sink
    # is live + the engine's VAD has settled before the first word. A Bluetooth
    # inject sink (e.g. the JBL) resuming from idle drops the start of a fresh
    # stream, so it needs the most; a wired sink still benefits (clean VAD onset).
    inject_lead_in_ms = 500

    def __init__(self, inject_sink: Optional[str] = None, inject_gain: int = 170):
        self._inject_sink = inject_sink
        # near-field (a dedicated user speaker by the mic) needs no boost; the
        # shared-speaker far-field case keeps the boost to clear the echo floor.
        self.inject_gain = 100 if inject_sink else inject_gain
        self._keepalive: Optional[subprocess.Popen] = None

    @property
    def inject_target(self) -> Optional[str]:
        return self._inject_sink

    @contextlib.contextmanager
    def session(self) -> Iterator["SpeakerAcoustics"]:
        # A dedicated inject sink suspends between clips (the assistant no longer
        # holds it open), and a Bluetooth sink then drops the start of the next
        # freshly-resumed stream -- truncating each injected clip's first word(s).
        # Hold it awake with a continuous silent stream for the whole run so every
        # clip plays from a warm link. (No-op for the shared-speaker case, where
        # the assistant's own output keeps the default sink warm.)
        if self._inject_sink:
            with contextlib.suppress(Exception):
                self._keepalive = subprocess.Popen(
                    ["pacat", f"--device={self._inject_sink}", "--channels=1",
                     "--rate=48000", "--format=s16le", "/dev/zero"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        try:
            yield self
        finally:
            if self._keepalive is not None:
                self._keepalive.terminate()
                with contextlib.suppress(Exception):
                    self._keepalive.wait(timeout=3)
                self._keepalive = None

    def route(self, pid: int) -> None:  # nothing to move
        pass

    def capture_ready(self, pid: int) -> bool:
        return True                # the engine already opened the real mic


def make_acoustics(mode: str, *, latency_ms: int = 260, inject_sink: Optional[str] = None):
    if mode == "cable":
        return CableAcoustics()
    if mode == "delay":
        return DelayAcoustics(latency_ms=latency_ms)
    if mode == "speaker":
        return SpeakerAcoustics(inject_sink=inject_sink)
    raise ValueError(f"unknown acoustics mode: {mode!r}")
