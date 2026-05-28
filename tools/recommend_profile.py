"""Probe the host hardware and recommend a ``device_profile`` from
``config.json``.

Run it once per machine to get a one-line suggestion you can pass to
``python -m core --device <name>``. The decision tree (cores / RAM /
GPU kind+VRAM) mirrors the latency table in
``docs/target_architecture.md`` so the recommendation matches the
modelled specsim numbers.

Stdlib-only by design -- no torch, no psutil, no nvidia-py. Probes:

* CPU cores via :func:`os.cpu_count`;
* RAM via ``/proc/meminfo`` (Linux), ``sysctl hw.memsize`` (macOS), or
  ``GlobalMemoryStatusEx`` (Windows);
* GPU via ``nvidia-smi`` if on PATH (NVIDIA), or by checking for an
  Apple Silicon platform.

Output is JSON-ish text so it can be piped or eyeballed. Exit code 0 on
success; 2 if no profile matched.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# --- detection -----------------------------------------------------------

@dataclass
class HostInfo:
    cores: int
    ram_gb: float
    gpu_kind: Optional[str]  # 'nvidia' | 'apple' | None
    gpu_mem_gb: float        # 0 if no discrete GPU; Apple unified = RAM
    mobile: bool = False     # Android / iOS heuristic; selects phone* profiles


def detect_cores() -> int:
    return os.cpu_count() or 1


def detect_ram_gb() -> float:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except OSError:
            pass
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=2,
            ).stdout.strip()
            return int(out) / (1024 ** 3)
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            pass
    if system == "Windows":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:  # noqa: BLE001
            pass
    return 0.0


def detect_nvidia_gpu_mem_gb() -> float:
    """Return the largest NVIDIA GPU's VRAM in GB, or 0 if none."""
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0.0
    best_mb = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            best_mb = max(best_mb, int(line))
        except ValueError:
            continue
    return best_mb / 1024 if best_mb else 0.0


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def is_mobile_os() -> bool:
    """True when we look like we're on Android or iOS. Distinguishes a 16 GB
    ARM laptop from a 16 GB Android phone -- otherwise the decision tree
    would mis-route a desktop box with low RAM to a llamacpp/GGUF profile."""
    if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"):
        return True
    if os.path.exists("/system/build.prop"):
        return True
    return False


def probe(*, ram_gb: Optional[float] = None,
          nvidia_mem_gb: Optional[float] = None,
          apple: Optional[bool] = None,
          cores: Optional[int] = None,
          mobile: Optional[bool] = None) -> HostInfo:
    """Gather a :class:`HostInfo`. Test code can inject every value;
    production code calls with no args and we probe everything."""
    cores = cores if cores is not None else detect_cores()
    ram = ram_gb if ram_gb is not None else detect_ram_gb()
    nvidia = nvidia_mem_gb if nvidia_mem_gb is not None else detect_nvidia_gpu_mem_gb()
    on_apple = apple if apple is not None else is_apple_silicon()
    on_mobile = mobile if mobile is not None else is_mobile_os()
    if nvidia >= 4:
        return HostInfo(cores=cores, ram_gb=ram, gpu_kind="nvidia", gpu_mem_gb=nvidia, mobile=on_mobile)
    if on_apple:
        return HostInfo(cores=cores, ram_gb=ram, gpu_kind="apple", gpu_mem_gb=ram, mobile=on_mobile)
    return HostInfo(cores=cores, ram_gb=ram, gpu_kind=None, gpu_mem_gb=0.0, mobile=on_mobile)


# --- recommendation ------------------------------------------------------

def recommend(info: HostInfo) -> tuple[str, str]:
    """Return ``(profile_name, rationale)``. Mirrors the device tiers in
    ``docs/target_architecture.md`` and the specsim model_fit thresholds.

    Order: a discrete NVIDIA/Apple GPU wins outright; otherwise the
    ``mobile`` flag (set by ``is_mobile_os()`` or injected) decides
    whether a low-RAM host is a phone or a small CPU laptop."""
    if info.gpu_kind == "nvidia" and info.gpu_mem_gb >= 16:
        return ("desktop_gpu_4090",
                "NVIDIA GPU >= 16 GB VRAM fits gemma3:12b comfortably; both input gates ON")
    if info.gpu_kind == "nvidia" and info.gpu_mem_gb >= 8:
        return ("desktop",
                "NVIDIA GPU 8-16 GB VRAM: gemma3:12b on the edge; use the 'desktop' default")
    if info.gpu_kind == "apple" and info.ram_gb >= 16:
        return ("macbook_m_series",
                "Apple Silicon with >= 16 GB unified memory; gemma3:4b on Metal + 1b fast tier")
    if info.mobile:
        if info.ram_gb >= 8:
            return ("phone",
                    "Android/iOS class, 8+ GB RAM: llamacpp + gemma3:4b GGUF main, 1b fast")
        return ("phone_lite",
                "Android/iOS class, < 8 GB RAM: single-tier gemma3:1b GGUF; cloud REQUIRED")
    # Desktop / laptop CPU path.
    if info.ram_gb >= 8:
        return ("cpu_laptop",
                "CPU-only laptop, 8+ GB RAM: gemma3:4b main; enable cloud hedge for paragraph answers")
    if info.ram_gb >= 4:
        return ("phone_lite",
                "< 8 GB RAM on a desktop OS: borrow the lite profile (gemma3:1b GGUF); cloud REQUIRED")
    return ("phone_lite",
            "very constrained host; the lite profile is the only realistic on-device option")


# --- CLI -----------------------------------------------------------------

def format_report(info: HostInfo, profile: str, rationale: str) -> str:
    gpu_desc = "none"
    if info.gpu_kind == "nvidia":
        gpu_desc = f"NVIDIA, {info.gpu_mem_gb:.1f} GB VRAM"
    elif info.gpu_kind == "apple":
        gpu_desc = f"Apple Silicon (unified {info.ram_gb:.1f} GB)"
    lines = [
        "Host probe:",
        f"  cores      : {info.cores}",
        f"  ram        : {info.ram_gb:.1f} GB",
        f"  gpu        : {gpu_desc}",
        "",
        f"Recommended profile: {profile}",
        f"Rationale          : {rationale}",
        "",
        f"Run with:  python -m core --device {profile}",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    info = probe()
    profile, rationale = recommend(info)
    print(format_report(info, profile, rationale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
