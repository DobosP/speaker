"""Native WASAPI communications-category capture (ADR-0019 unblock; ADR-0082).

Opens the DEFAULT capture endpoint through ``IAudioClient2`` with
``AudioClientProperties.eCategory = AudioCategory_Communications`` -- the
"Teams path" ADR-0013 assumed -- and, critically, VERIFIES post-``Initialize``
that the OS audio-effects pipeline actually attached an Acoustic Echo
Cancellation effect (``IAudioEffectsManager``, Windows 11 build 22000+).

Why verification is the point: Windows maps the app-requested *category* to a
driver-defined *processing mode*, and "if a particular mode is not supported by
the driver, Windows will use the next best matching mode" -- SILENTLY. A
successfully-tagged Communications stream on a driver without a Communications
APO gets no AEC at all. Construct-success (what the pre-ADR-0019 code treated
as "applied") therefore proves nothing; the machine-checkable contract is the
post-open effects list reporting ``ACOUSTIC_ECHO_CANCELLATION`` present -- the
Windows analogue of the Linux ``pactl`` echo-cancel node probe in
``core/readiness.py``.

Threading: ALL COM work is confined to one dedicated MTA reader thread (build
chain, Start/Stop, packet pump). The engine-facing surface is duck-typed to
``_RecoveringInputStream``'s candidate contract -- ``.samplerate``/``.device``
attrs, ``.start()``, ``.read(frames) -> (ndarray, overflowed)``, ``.close()``,
``.abort()`` -- and raises ``sounddevice.PortAudioError``-SHAPED errors
(message, int code in ``args[1]``) so the existing TRANSIENT/REOPEN recovery
classifier works unchanged.

Every GUID/vtable below was verified against the win32metadata-generated
``windows`` crate 0.61.1 (NOT the MS docs method tables, which are alphabetical
-- IAudioEffectsManager's real vtable is Register/Unregister/Get/Set).
"""
from __future__ import annotations

import ctypes
import struct as _struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --- constants (win32metadata-verified) --------------------------------------
CLSID_MM_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
IID_IMM_DEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
IID_IMM_DEVICE = "{D666063F-1587-4E43-81F1-B948E807363F}"
IID_IAUDIO_CLIENT = "{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}"
IID_IAUDIO_CLIENT2 = "{726778CD-F60A-4EDA-82DE-E47610CD78AA}"
IID_IAUDIO_CAPTURE_CLIENT = "{C8ADBD64-E71E-48A0-A4DE-185C395CD317}"
IID_IAUDIO_EFFECTS_MANAGER = "{4460B3AE-4B44-4527-8676-7548A8ACD260}"

# ksmedia.h effect-type GUIDs (Win11 effects framework).
GUID_EFFECT_AEC = "{6F64ADBE-8211-11E2-8C70-2C27D7F001FA}"
GUID_EFFECT_NS = "{6F64ADBF-8211-11E2-8C70-2C27D7F001FA}"
GUID_EFFECT_AGC = "{6F64ADC0-8211-11E2-8C70-2C27D7F001FA}"

E_CAPTURE = 1          # EDataFlow eCapture
E_ROLE_COMMUNICATIONS = 2  # ERole eCommunications
AUDIO_CATEGORY_COMMUNICATIONS = 3  # AUDIO_STREAM_CATEGORY (capture-legal set:
# Communications/Speech/Other ONLY -- everything else is render-only)
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDIO_EFFECT_STATE_OFF = 0
AUDIO_EFFECT_STATE_ON = 1
AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY = 0x2
AUDCLNT_E_DEVICE_INVALIDATED = -2004287484  # 0x88890004 as signed HRESULT

_REFTIMES_PER_SEC = 10_000_000  # 100ns units
_BUFFER_DURATION_100NS = 2_000_000  # 200 ms shared-mode buffer

# PortAudio codes the recovery classifier understands (core/engines/
# _recovering_input.py TRANSIENT_CODES/REOPEN_CODES read exc.args[1]).
_PA_UNANTICIPATED_HOST_ERROR = -9999  # REOPEN class
_PA_DEVICE_UNAVAILABLE = -9985        # REOPEN class

_MIN_WIN11_BUILD = 22000


def _pa_error(message: str, code: int) -> Exception:
    """A sounddevice.PortAudioError-shaped exception (args = (msg, code)).

    Lazy import: this module must import cleanly on Linux/CI where the error
    type still exists (sounddevice is a hard dep of the sherpa engine) but we
    never want an import-time hard bind for the pure-logic tests."""
    try:
        import sounddevice as sd

        return sd.PortAudioError(message, code)
    except Exception:  # noqa: BLE001 - tests without sounddevice: same shape
        err = RuntimeError(message, code)
        return err


# --- pure logic (platform-free, unit-tested) ---------------------------------

KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = "00000003-0000-0010-8000-00aa00389b71"
KSDATAFORMAT_SUBTYPE_PCM = "00000001-0000-0010-8000-00aa00389b71"
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass(frozen=True)
class MixFormat:
    sample_rate: int
    channels: int
    bits: int
    is_float: bool


def parse_waveformatex(raw: bytes) -> MixFormat:
    """Parse a WAVEFORMATEX(TENSIBLE) blob into the fields we consume.

    Raises ValueError on anything we cannot faithfully convert -- the caller
    fails the open rather than guessing at sample layout (garbled capture is
    the one outcome worse than no capture; see the Phase-0 mic post-mortem)."""
    if len(raw) < 16:
        raise ValueError("WAVEFORMATEX blob too short")
    tag, channels, rate, _avg, _align, bits = _struct.unpack("<HHIIHH", raw[:16])
    is_float = tag == WAVE_FORMAT_IEEE_FLOAT
    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(raw) < 40:
            raise ValueError("WAVEFORMATEXTENSIBLE blob too short")
        sub = raw[24:40]
        d1, d2, d3 = _struct.unpack("<IHH", sub[:8])
        guid = f"{d1:08x}-{d2:04x}-{d3:04x}-" + sub[8:10].hex() + "-" + sub[10:16].hex()
        if guid == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT:
            is_float = True
        elif guid == KSDATAFORMAT_SUBTYPE_PCM:
            is_float = False
        else:
            raise ValueError(f"unsupported WASAPI subformat {guid}")
    elif tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise ValueError(f"unsupported WAVEFORMATEX tag 0x{tag:04x}")
    if is_float and bits != 32:
        raise ValueError(f"float mix format with {bits} bits")
    if not is_float and bits not in (16, 24, 32):
        raise ValueError(f"PCM mix format with {bits} bits")
    if channels < 1 or rate < 8000:
        raise ValueError(f"implausible mix format: {channels}ch @ {rate}Hz")
    return MixFormat(int(rate), int(channels), int(bits), bool(is_float))


def downmix_to_mono_float32(raw: bytes, fmt: MixFormat) -> np.ndarray:
    """Interleaved capture bytes -> mono float32 in [-1, 1] (channel average)."""
    if fmt.is_float:
        x = np.frombuffer(raw, dtype="<f4")
    elif fmt.bits == 16:
        x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    elif fmt.bits == 32:
        x = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
    else:  # 24-bit packed
        b = np.frombuffer(raw, dtype="u1").reshape(-1, 3)
        as32 = (
            b[:, 0].astype("i4")
            | (b[:, 1].astype("i4") << 8)
            | (b[:, 2].astype("i4") << 16)
        )
        as32 = np.where(as32 & 0x800000, as32 - 0x1000000, as32)
        x = as32.astype("float32") / 8388608.0
    if fmt.channels > 1:
        n = (x.size // fmt.channels) * fmt.channels
        x = x[:n].reshape(-1, fmt.channels).mean(axis=1)
    return np.ascontiguousarray(x, dtype="float32")


def effects_verdict(effects: list[dict]) -> dict:
    """The route verdict from a GetAudioEffects snapshot.

    ``aec_active`` is THE contract bit: the word-cut route is verified only
    when the OS reports an AEC effect present and ON for this stream."""
    def _on(guid: str) -> bool:
        g = guid.strip("{}").lower()
        return any(
            e.get("id", "").strip("{}").lower() == g
            and int(e.get("state", AUDIO_EFFECT_STATE_OFF)) == AUDIO_EFFECT_STATE_ON
            for e in effects
        )

    return {
        "aec_active": _on(GUID_EFFECT_AEC),
        "ns_active": _on(GUID_EFFECT_NS) ,
        "agc_active": _on(GUID_EFFECT_AGC),
        "effect_count": len(effects),
        "effects": effects,
    }


class SampleRing:
    """Bounded mono-float32 ring between the COM pump and blocking read().

    Plain deque + Condition; drops the OLDEST audio on overflow and flags it so
    read() reports ``overflowed=True`` exactly like sounddevice does."""

    def __init__(self, max_seconds: float, sample_rate: int):
        self._max = max(1, int(max_seconds * sample_rate))
        self._chunks: deque[np.ndarray] = deque()
        self._count = 0
        self._overflowed = False
        self._closed = False
        self._cond = threading.Condition()

    def put(self, samples: np.ndarray, *, discontinuity: bool = False) -> None:
        with self._cond:
            if discontinuity:
                self._overflowed = True
            self._chunks.append(samples)
            self._count += samples.size
            while self._count > self._max and len(self._chunks) > 1:
                old = self._chunks.popleft()
                self._count -= old.size
                self._overflowed = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def take(
        self,
        frames: int,
        *,
        timeout: float,
        fatal: Callable[[], Optional[Exception]],
    ) -> tuple[np.ndarray, bool]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._count < frames:
                exc = fatal()
                if exc is not None:
                    raise exc
                if self._closed:
                    raise _pa_error(
                        "wasapi-comm stream closed during read",
                        _PA_DEVICE_UNAVAILABLE,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _pa_error(
                        "wasapi-comm read timed out (device stalled?)",
                        _PA_UNANTICIPATED_HOST_ERROR,
                    )
                self._cond.wait(min(remaining, 0.1))
            out = np.empty(frames, dtype="float32")
            filled = 0
            while filled < frames:
                head = self._chunks[0]
                need = frames - filled
                if head.size <= need:
                    out[filled : filled + head.size] = head
                    filled += head.size
                    self._chunks.popleft()
                else:
                    out[filled:frames] = head[:need]
                    self._chunks[0] = head[need:]
                    filled = frames
            self._count -= frames
            overflowed, self._overflowed = self._overflowed, False
            return out.reshape(-1, 1), overflowed


# --- COM layer (win32 only; all COM confined to the pump thread) -------------

_com_lock = threading.Lock()
_com_ns: Optional[dict] = None


def _declare_com() -> dict:
    """Declare the comtypes interfaces once (vtable order metadata-verified)."""
    global _com_ns
    with _com_lock:
        if _com_ns is not None:
            return _com_ns
        import comtypes
        from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

        wintypes = ctypes.wintypes

        class AudioClientProperties(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint32),
                ("bIsOffload", wintypes.BOOL),
                ("eCategory", ctypes.c_int),
                ("Options", ctypes.c_int),
            ]

        class AUDIO_EFFECT(ctypes.Structure):
            _fields_ = [
                ("id", GUID),
                ("canSetState", wintypes.BOOL),
                ("state", ctypes.c_int),
            ]

        class IAudioClient(IUnknown):
            _iid_ = GUID(IID_IAUDIO_CLIENT)
            _methods_ = [
                COMMETHOD([], HRESULT, "Initialize",
                          (["in"], ctypes.c_int, "ShareMode"),
                          (["in"], ctypes.c_uint32, "StreamFlags"),
                          (["in"], ctypes.c_longlong, "hnsBufferDuration"),
                          (["in"], ctypes.c_longlong, "hnsPeriodicity"),
                          (["in"], ctypes.c_void_p, "pFormat"),
                          (["in"], ctypes.c_void_p, "AudioSessionGuid")),
                COMMETHOD([], HRESULT, "GetBufferSize",
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pNumBufferFrames")),
                COMMETHOD([], HRESULT, "GetStreamLatency",
                          (["out"], ctypes.POINTER(ctypes.c_longlong), "phnsLatency")),
                COMMETHOD([], HRESULT, "GetCurrentPadding",
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pNumPaddingFrames")),
                COMMETHOD([], HRESULT, "IsFormatSupported",
                          (["in"], ctypes.c_int, "ShareMode"),
                          (["in"], ctypes.c_void_p, "pFormat"),
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppClosestMatch")),
                COMMETHOD([], HRESULT, "GetMixFormat",
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppDeviceFormat")),
                COMMETHOD([], HRESULT, "GetDevicePeriod",
                          (["out"], ctypes.POINTER(ctypes.c_longlong), "phnsDefaultDevicePeriod"),
                          (["out"], ctypes.POINTER(ctypes.c_longlong), "phnsMinimumDevicePeriod")),
                COMMETHOD([], HRESULT, "Start"),
                COMMETHOD([], HRESULT, "Stop"),
                COMMETHOD([], HRESULT, "Reset"),
                COMMETHOD([], HRESULT, "SetEventHandle",
                          (["in"], wintypes.HANDLE, "eventHandle")),
                COMMETHOD([], HRESULT, "GetService",
                          (["in"], ctypes.POINTER(GUID), "riid"),
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppv")),
            ]

        class IAudioClient2(IAudioClient):
            _iid_ = GUID(IID_IAUDIO_CLIENT2)
            _methods_ = [
                COMMETHOD([], HRESULT, "IsOffloadCapable",
                          (["in"], ctypes.c_int, "Category"),
                          (["out"], ctypes.POINTER(wintypes.BOOL), "pbOffloadCapable")),
                COMMETHOD([], HRESULT, "SetClientProperties",
                          (["in"], ctypes.POINTER(AudioClientProperties), "pProperties")),
                COMMETHOD([], HRESULT, "GetBufferSizeLimits",
                          (["in"], ctypes.c_void_p, "pFormat"),
                          (["in"], wintypes.BOOL, "bEventDriven"),
                          (["out"], ctypes.POINTER(ctypes.c_longlong), "phnsMinBufferDuration"),
                          (["out"], ctypes.POINTER(ctypes.c_longlong), "phnsMaxBufferDuration")),
            ]

        class IAudioCaptureClient(IUnknown):
            _iid_ = GUID(IID_IAUDIO_CAPTURE_CLIENT)
            _methods_ = [
                COMMETHOD([], HRESULT, "GetBuffer",
                          (["out"], ctypes.POINTER(ctypes.POINTER(ctypes.c_byte)), "ppData"),
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pNumFramesToRead"),
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pdwFlags"),
                          (["in"], ctypes.c_void_p, "pu64DevicePosition"),
                          (["in"], ctypes.c_void_p, "pu64QPCPosition")),
                COMMETHOD([], HRESULT, "ReleaseBuffer",
                          (["in"], ctypes.c_uint32, "NumFramesRead")),
                COMMETHOD([], HRESULT, "GetNextPacketSize",
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pNumFramesInNextPacket")),
            ]

        # Vtable order metadata-verified: Register/Unregister/Get/Set (the MS
        # docs method table is ALPHABETICAL -- Get first there -- do not "fix"
        # this back from the docs).
        class IAudioEffectsManager(IUnknown):
            _iid_ = GUID(IID_IAUDIO_EFFECTS_MANAGER)
            _methods_ = [
                COMMETHOD([], HRESULT, "RegisterAudioEffectsChangedNotificationCallback",
                          (["in"], ctypes.c_void_p, "client")),
                COMMETHOD([], HRESULT, "UnregisterAudioEffectsChangedNotificationCallback",
                          (["in"], ctypes.c_void_p, "client")),
                COMMETHOD([], HRESULT, "GetAudioEffects",
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "effects"),
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "numeffects")),
                COMMETHOD([], HRESULT, "SetAudioEffectState",
                          (["in"], GUID, "effectId"),
                          (["in"], ctypes.c_int, "state")),
            ]

        class IMMDevice(IUnknown):
            _iid_ = GUID(IID_IMM_DEVICE)
            _methods_ = [
                COMMETHOD([], HRESULT, "Activate",
                          (["in"], ctypes.POINTER(GUID), "iid"),
                          (["in"], ctypes.c_uint32, "dwClsCtx"),
                          (["in"], ctypes.c_void_p, "pActivationParams"),
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppInterface")),
                COMMETHOD([], HRESULT, "OpenPropertyStore",
                          (["in"], ctypes.c_uint32, "stgmAccess"),
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppProperties")),
                COMMETHOD([], HRESULT, "GetId",
                          (["out"], ctypes.POINTER(ctypes.c_wchar_p), "ppstrId")),
                COMMETHOD([], HRESULT, "GetState",
                          (["out"], ctypes.POINTER(ctypes.c_uint32), "pdwState")),
            ]

        class IMMDeviceEnumerator(IUnknown):
            _iid_ = GUID(IID_IMM_DEVICE_ENUMERATOR)
            _methods_ = [
                COMMETHOD([], HRESULT, "EnumAudioEndpoints",
                          (["in"], ctypes.c_int, "dataFlow"),
                          (["in"], ctypes.c_uint32, "dwStateMask"),
                          (["out"], ctypes.POINTER(ctypes.c_void_p), "ppDevices")),
                COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint",
                          (["in"], ctypes.c_int, "dataFlow"),
                          (["in"], ctypes.c_int, "role"),
                          (["out"], ctypes.POINTER(ctypes.POINTER(IMMDevice)), "ppEndpoint")),
                COMMETHOD([], HRESULT, "GetDevice",
                          (["in"], ctypes.c_wchar_p, "pwstrId"),
                          (["out"], ctypes.POINTER(ctypes.POINTER(IMMDevice)), "ppDevice")),
                COMMETHOD([], HRESULT, "RegisterEndpointNotificationCallback",
                          (["in"], ctypes.c_void_p, "pClient")),
                COMMETHOD([], HRESULT, "UnregisterEndpointNotificationCallback",
                          (["in"], ctypes.c_void_p, "pClient")),
            ]

        _com_ns = {
            "comtypes": comtypes,
            "GUID": GUID,
            "AudioClientProperties": AudioClientProperties,
            "AUDIO_EFFECT": AUDIO_EFFECT,
            "IAudioClient": IAudioClient,
            "IAudioClient2": IAudioClient2,
            "IAudioCaptureClient": IAudioCaptureClient,
            "IAudioEffectsManager": IAudioEffectsManager,
            "IMMDevice": IMMDevice,
            "IMMDeviceEnumerator": IMMDeviceEnumerator,
        }
        return _com_ns


def _as_void_p(ptr) -> ctypes.c_void_p:
    """Pointer-width wrap for CoTaskMemFree regardless of how comtypes
    marshalled the [out] param (bare int today; a ctypes pointer object under
    other comtypes versions). Never silently substitutes NULL for a live
    allocation -- that would be an invisible per-open native leak."""
    if ptr is None:
        return ctypes.c_void_p(None)
    if isinstance(ptr, int):
        return ctypes.c_void_p(ptr)
    if isinstance(ptr, ctypes.c_void_p):
        return ptr
    return ctypes.cast(ptr, ctypes.c_void_p)


_kernel32 = None


def _get_kernel32():
    """kernel32 with explicit pointer-width signatures (a bare ctypes.windll
    call marshals HANDLEs through 32-bit c_int on a 64-bit interpreter)."""
    global _kernel32
    if _kernel32 is None:
        wt = ctypes.wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateEventW.restype = wt.HANDLE
        k.CreateEventW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR]
        k.WaitForSingleObject.restype = wt.DWORD
        k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        k.CloseHandle.restype = wt.BOOL
        k.CloseHandle.argtypes = [wt.HANDLE]
        _kernel32 = k
    return _kernel32


def _windows_build() -> int:
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - non-Windows
        return 0


class WasapiCommCapture:
    """Communications-category capture client (default endpoint, v1).

    Duck-typed to the ``_RecoveringInputStream`` candidate contract. The
    requested sample rate is a HINT only: shared-mode WASAPI runs at the mix
    rate; ``.samplerate`` reports the true rate and the engine's existing
    resampler chain bridges to the model rate (same pattern as any device
    that rejects the preferred rate)."""

    def __init__(self, device=None, samplerate: int = 0, *, ring_seconds: float = 4.0):
        if not sys.platform.startswith("win"):
            raise _pa_error("wasapi-comm capture is Windows-only", _PA_DEVICE_UNAVAILABLE)
        if device is not None:
            # v1 is default-endpoint only: a named selector silently binding to
            # the WRONG mic would poison enrollment/calibration domains.
            raise _pa_error(
                "wasapi-comm capture supports only the default input device "
                f"(got selector {device!r})",
                _PA_DEVICE_UNAVAILABLE,
            )
        if _windows_build() < _MIN_WIN11_BUILD:
            # No effects framework => no verifiable AEC contract. Fail closed
            # per ADR-0019 rather than trusting an unprovable category tag.
            raise _pa_error(
                "wasapi-comm capture requires Windows 11 build 22000+ for the "
                "audio-effects verification framework",
                _PA_DEVICE_UNAVAILABLE,
            )
        self.device = device
        self.samplerate: int = 0  # set by the pump thread from the mix format
        self.channels = 1
        self.mix_format: Optional[MixFormat] = None
        self.effects: list[dict] = []
        self.effects_error: str = ""
        self.verdict: dict = {}
        self._ring: Optional[SampleRing] = None
        self._ring_seconds = float(ring_seconds)
        self._fatal: Optional[Exception] = None
        self._ready = threading.Event()
        self._start_req = threading.Event()
        self._started = threading.Event()
        self._stop_req = threading.Event()
        self._thread = threading.Thread(
            target=self._pump, name="wasapi-comm-capture", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=8.0):
            self._stop_req.set()
            raise _pa_error(
                "wasapi-comm open timed out", _PA_UNANTICIPATED_HOST_ERROR
            )
        if self._fatal is not None:
            raise self._fatal

    # --- candidate contract ---------------------------------------------------

    def start(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        self._start_req.set()
        if not self._started.wait(timeout=4.0):
            exc = self._fatal or _pa_error(
                "wasapi-comm Start timed out", _PA_UNANTICIPATED_HOST_ERROR
            )
            raise exc
        if self._fatal is not None:
            raise self._fatal

    def read(self, frames: int):
        ring = self._ring
        if ring is None:
            raise _pa_error("wasapi-comm stream not open", _PA_DEVICE_UNAVAILABLE)
        # Generous timeout: frames/rate plus device-stall headroom; the
        # recovery wrapper treats the raised REOPEN-class error as its signal.
        timeout = max(1.0, 2.0 * frames / max(self.samplerate, 1) + 1.0)
        return ring.take(frames, timeout=timeout, fatal=lambda: self._fatal)

    def _halt(self) -> None:
        """The single halt sequence stop()/close()/abort() share."""
        self._stop_req.set()
        if self._ring is not None:
            self._ring.close()

    def stop(self) -> None:
        """Halt audio delivery (recovery-wrapper teardown calls stop() then
        close()). Restart is NOT supported -- the wrapper reopens via its
        opener on recovery, never by restarting a stopped candidate."""
        self._halt()

    def close(self) -> None:
        self._halt()
        self._thread.join(timeout=3.0)

    def abort(self) -> None:
        self._halt()
        # no join: abort must return promptly (mirrors sounddevice semantics)

    # --- the single COM thread ------------------------------------------------

    def _fail(self, exc: Exception) -> None:
        if self._fatal is None:
            self._fatal = exc
        self._ready.set()
        self._started.set()
        ring = self._ring
        if ring is not None:
            ring.close()

    def _pump(self) -> None:
        try:
            ns = _declare_com()
        except Exception as exc:  # noqa: BLE001 - comtypes missing/broken
            self._fail(_pa_error(f"comtypes unavailable: {exc}", _PA_DEVICE_UNAVAILABLE))
            return
        comtypes = ns["comtypes"]
        GUID = ns["GUID"]
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            # RPC_E_CHANGED_MODE: thread already STA -- acceptable, proceed.
            pass
        kernel32 = _get_kernel32()
        event = kernel32.CreateEventW(None, False, False, None)
        if not event:
            self._fail(_pa_error(
                "CreateEventW failed for the capture event",
                _PA_UNANTICIPATED_HOST_ERROR,
            ))
            try:
                comtypes.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass
            return
        enum = None
        dev = None
        client = None
        capture = None
        mix_ptr = None
        try:
            enum = comtypes.CoCreateInstance(
                GUID(CLSID_MM_DEVICE_ENUMERATOR),
                interface=ns["IMMDeviceEnumerator"],
                clsctx=comtypes.CLSCTX_INPROC_SERVER,
            )
            dev = enum.GetDefaultAudioEndpoint(E_CAPTURE, E_ROLE_COMMUNICATIONS)
            # comtypes returns [out] params as values -- do NOT pass receivers.
            raw = dev.Activate(
                ctypes.byref(GUID(IID_IAUDIO_CLIENT2)),
                comtypes.CLSCTX_INPROC_SERVER,
                None,
            )
            client = ctypes.cast(
                ctypes.c_void_p(raw if isinstance(raw, int) else raw),
                ctypes.POINTER(ns["IAudioClient2"]),
            )
            # ORDER MATTERS: SetClientProperties BEFORE GetMixFormat/Initialize
            # -- the category can change the answers (MS guidance).
            props = ns["AudioClientProperties"](
                cbSize=ctypes.sizeof(ns["AudioClientProperties"]),
                bIsOffload=False,
                eCategory=AUDIO_CATEGORY_COMMUNICATIONS,
                Options=0,
            )
            client.SetClientProperties(ctypes.byref(props))
            mix_ptr = client.GetMixFormat()
            if not mix_ptr:
                raise RuntimeError("GetMixFormat returned NULL")
            head = ctypes.string_at(mix_ptr, 18)
            cb = _struct.unpack("<H", head[16:18])[0]
            fmt_bytes = ctypes.string_at(mix_ptr, 18 + cb)
            fmt = parse_waveformatex(fmt_bytes)
            self.mix_format = fmt
            self.samplerate = fmt.sample_rate
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                _BUFFER_DURATION_100NS,
                0,
                mix_ptr,
                None,
            )
            client.SetEventHandle(event)
            svc = client.GetService(ctypes.byref(GUID(IID_IAUDIO_CAPTURE_CLIENT)))
            capture = ctypes.cast(
                ctypes.c_void_p(svc if isinstance(svc, int) else svc),
                ctypes.POINTER(ns["IAudioCaptureClient"]),
            )
            self.effects = self._snapshot_effects(ns, client)
            self.verdict = effects_verdict(self.effects)
            self._ring = SampleRing(self._ring_seconds, fmt.sample_rate)
            self._ready.set()

            bytes_per_frame = fmt.channels * (fmt.bits // 8)
            started = False
            while not self._stop_req.is_set():
                if not started:
                    if self._start_req.wait(timeout=0.05):
                        client.Start()
                        started = True
                        self._started.set()
                    continue
                kernel32.WaitForSingleObject(event, 100)
                while True:
                    packet = capture.GetNextPacketSize()
                    if not packet:
                        break
                    # comtypes returns the three [out] params as a tuple.
                    data, frames, flags = capture.GetBuffer(None, None)
                    n = int(frames)
                    if n:
                        blob = ctypes.string_at(data, n * bytes_per_frame)
                        mono = downmix_to_mono_float32(blob, fmt)
                        disc = bool(int(flags) & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY)
                        self._ring.put(mono, discontinuity=disc)
                    capture.ReleaseBuffer(n)
            if started:
                try:
                    client.Stop()
                except Exception:  # noqa: BLE001 - device may be gone
                    pass
        except Exception as exc:  # noqa: BLE001 - map to REOPEN-class error
            hres = getattr(exc, "hresult", None)
            code = (
                _PA_DEVICE_UNAVAILABLE
                if hres == AUDCLNT_E_DEVICE_INVALIDATED
                else _PA_UNANTICIPATED_HOST_ERROR
            )
            self._fail(_pa_error(f"wasapi-comm capture failed: {exc}", code))
        finally:
            if mix_ptr:
                # CoTaskMemFree takes a pointer-width arg; a bare Python int
                # over 2**31 overflows the default c_int marshalling.
                ctypes.windll.ole32.CoTaskMemFree(_as_void_p(mix_ptr))
            # Release every COM proxy BEFORE CoUninitialize: rebinding the
            # only reference triggers comtypes' Release() now, while the MTA
            # is still initialized on this thread. Leaving them to die at
            # frame teardown would Release() into an uninitialized apartment
            # (undefined per COM) on every reopen/recovery cycle.
            capture = None  # noqa: F841 - deliberate release ordering
            client = None  # noqa: F841
            dev = None  # noqa: F841
            enum = None  # noqa: F841
            if event:
                kernel32.CloseHandle(event)
            try:
                comtypes.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass

    def _snapshot_effects(self, ns, client) -> list[dict]:
        """GetAudioEffects via IAudioEffectsManager; empty list when absent."""
        GUID = ns["GUID"]
        try:
            svc = client.GetService(ctypes.byref(GUID(IID_IAUDIO_EFFECTS_MANAGER)))
            mgr = ctypes.cast(
                ctypes.c_void_p(svc if isinstance(svc, int) else svc),
                ctypes.POINTER(ns["IAudioEffectsManager"]),
            )
            arr_ptr, count = mgr.GetAudioEffects()
            effects: list[dict] = []
            n = int(count)
            if arr_ptr and n:
                eff_arr = ctypes.cast(
                    arr_ptr, ctypes.POINTER(ns["AUDIO_EFFECT"] * n)
                ).contents
                for e in eff_arr:
                    effects.append(
                        {
                            "id": str(e.id),
                            "canSetState": bool(e.canSetState),
                            "state": int(e.state),
                        }
                    )
                ctypes.windll.ole32.CoTaskMemFree(_as_void_p(arr_ptr))
            return effects
        except Exception as exc:  # noqa: BLE001 - manager missing => no proof
            # Distinguish "no effects" (empty list, call succeeded) from "the
            # effects API itself failed" -- the verdict consumer surfaces this.
            self.effects_error = f"{type(exc).__name__}: {exc}"
            return []


def probe_comm_capture(device=None) -> dict:
    """Short-lived open + effects snapshot: the readiness/enroll probe.

    Returns a dict verdict; NEVER raises (readiness renders the failure)."""
    try:
        cap = WasapiCommCapture(device=device)
    except Exception as exc:  # noqa: BLE001 - verdict, not crash
        return {
            "available": False,
            "aec_active": False,
            "error": f"{exc}",
            "build": _windows_build(),
        }
    try:
        out = {
            "available": True,
            "build": _windows_build(),
            "sample_rate": cap.samplerate,
            "mix_format": (
                f"{cap.mix_format.channels}ch/{cap.mix_format.bits}bit"
                f"{'f' if cap.mix_format.is_float else 'i'}@{cap.samplerate}"
                if cap.mix_format
                else "?"
            ),
        }
        out.update(cap.verdict)
        if cap.effects_error:
            out["effects_error"] = cap.effects_error
        return out
    finally:
        cap.close()
