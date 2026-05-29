"""Compute telemetry: CPU / RAM / GPU usage, system-wide and for this process.

A :func:`snapshot` captures a point-in-time reading; :class:`SystemMonitor`
takes a baseline, samples periodically on a background thread (so the hot path
is untouched), supports named marks (e.g. before/after model load), and at stop
folds baseline / final / peak / per-mark / deltas into the run summary.

Graceful when ``psutil`` (CPU/RAM/process) or ``nvidia-smi`` (GPU) are absent:
the missing fields are just omitted. Note that on the desktop path the LLM runs
in the *Ollama* process, so GPU memory/util here is system-wide (it reflects
Ollama), while ``proc_*`` is this Python process (sherpa STT/TTS + runtime).

Quick standalone reading:  ``python -m core.sysinfo``
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("speaker.sysinfo")

_PEAK_PROC_KEYS = ("cpu_percent", "ram_used_mb", "ram_percent", "proc_rss_mb", "proc_cpu_percent")


def _gpu_snapshot(run: Callable = subprocess.run) -> Optional[list]:
    try:
        r = run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:  # noqa: BLE001 - no nvidia-smi / not on PATH
        return None
    if getattr(r, "returncode", 1) != 0:
        return None
    gpus = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                {
                    "util_percent": float(parts[0]),
                    "mem_used_mb": float(parts[1]),
                    "mem_total_mb": float(parts[2]),
                    "temp_c": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return gpus or None


def snapshot() -> dict:
    """A point-in-time CPU/RAM/GPU reading (best-effort, never raises)."""
    out: dict = {"t": round(time.time(), 3)}
    try:
        import psutil  # noqa: PLC0415

        out["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = round(vm.used / 1e6, 1)
        out["ram_total_mb"] = round(vm.total / 1e6, 1)
        out["ram_percent"] = vm.percent
        p = psutil.Process()
        out["proc_rss_mb"] = round(p.memory_info().rss / 1e6, 1)
        out["proc_cpu_percent"] = p.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001 - psutil missing or restricted
        out["note"] = "psutil unavailable (pip install psutil) -- CPU/RAM omitted"
    out["gpu"] = _gpu_snapshot()  # None when no nvidia-smi
    return out


def _format(s: dict) -> str:
    bits = []
    if "cpu_percent" in s:
        bits.append(f"cpu={s['cpu_percent']}%")
    if "ram_used_mb" in s:
        bits.append(f"ram={s['ram_used_mb']}MB({s.get('ram_percent')}%)")
    if "proc_rss_mb" in s:
        bits.append(f"proc_rss={s['proc_rss_mb']}MB")
    g = (s.get("gpu") or [None])[0]
    if isinstance(g, dict):
        bits.append(f"gpu={g['util_percent']}% gpu_mem={g['mem_used_mb']}/{g['mem_total_mb']}MB")
    return " ".join(bits) or s.get("note", "no telemetry")


def _deltas(before: Optional[dict], after: Optional[dict]) -> dict:
    if not before or not after:
        return {}
    d = {}
    for k in ("ram_used_mb", "proc_rss_mb"):
        b, a = before.get(k), after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            d[f"{k}_delta"] = round(a - b, 1)
    gb = (before.get("gpu") or [None])[0]
    ga = (after.get("gpu") or [None])[0]
    if isinstance(gb, dict) and isinstance(ga, dict):
        d["gpu_mem_used_mb_delta"] = round(ga["mem_used_mb"] - gb["mem_used_mb"], 1)
    return d


class SystemMonitor:
    """Background sampler that records baseline/peak/final compute usage."""

    def __init__(self, summary=None, *, interval: float = 10.0, sampler: Callable[[], dict] = snapshot):
        self._summary = summary
        self.interval = interval
        self._sampler = sampler
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline: Optional[dict] = None
        self.last: Optional[dict] = None
        self.peak: dict = {}
        self.marks: dict = {}
        self.samples = 0

    def start(self) -> None:
        self.baseline = self._sampler()
        self._observe(self.baseline)
        log.info("system baseline: %s", _format(self.baseline))
        self._thread = threading.Thread(target=self._run, name="sysmon", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            s = self._sampler()
            self.samples += 1
            self._observe(s)
            log.debug("system sample: %s", _format(s))

    def _observe(self, s: dict) -> None:
        self.last = s
        for k in _PEAK_PROC_KEYS:
            v = s.get(k)
            if isinstance(v, (int, float)):
                self.peak[k] = max(self.peak.get(k, v), v)
        g = (s.get("gpu") or [None])[0]
        if isinstance(g, dict):
            for k in ("util_percent", "mem_used_mb"):
                v = g.get(k)
                if isinstance(v, (int, float)):
                    pk = f"gpu_{k}"
                    self.peak[pk] = max(self.peak.get(pk, v), v)

    def load_fraction(self) -> Optional[float]:
        """A cheap 0..1 system-load snapshot from the LAST background sample.

        Reads the most recent sample the background thread already took (never
        samples on the hot path) and returns the max of CPU% and GPU util%
        scaled to ``[0, 1]``. ``None`` when no telemetry is available (psutil +
        nvidia-smi both absent, or before the first sample) -- the router's
        ``live_nudge`` treats a ``None``/garbage load as no nudge, so a missing
        signal can never starve the local tier (the live-routing follow-up).

        This is the cheapest honest signal: it reuses the sampler's existing
        cadence rather than adding a synchronous CPU/GPU read to a turn."""
        s = self.last
        if not isinstance(s, dict):
            return None
        candidates: list[float] = []
        cpu = s.get("cpu_percent")
        if isinstance(cpu, (int, float)):
            candidates.append(float(cpu))
        gpu = s.get("gpu")
        g = gpu[0] if isinstance(gpu, list) and gpu else None
        if isinstance(g, dict):
            util = g.get("util_percent")
            if isinstance(util, (int, float)):
                candidates.append(float(util))
        if not candidates:
            return None
        return max(0.0, min(1.0, max(candidates) / 100.0))

    def mark(self, name: str) -> dict:
        """Snapshot at a named point (e.g. 'after_build') -> kept in the summary."""
        s = self._sampler()
        self.marks[name] = s
        self._observe(s)
        log.info("system mark %s: %s", name, _format(s))
        return s

    def _build(self, final: dict) -> dict:
        return {
            "baseline": self.baseline,
            "final": final,
            "peak": dict(self.peak),
            "marks": self.marks,
            "samples": self.samples,
            "deltas": _deltas(self.baseline, final),
        }

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
        final = self._sampler()
        self._observe(final)
        block = self._build(final)
        log.info(
            "system final: %s | deltas: %s", _format(final), block["deltas"] or "n/a"
        )
        if self._summary is not None:
            self._summary.system = block
        return block


if __name__ == "__main__":  # pragma: no cover - quick standalone reading
    import json

    print(json.dumps(snapshot(), indent=2))
