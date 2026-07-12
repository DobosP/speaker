"""Tests for the OTA autotest rig's gain-pinner (tools/autotest/ota_setup.py).

Pure: subprocess is faked, no audio devices touched. The regression under test is
that `gain_pinner` SAVES the pre-run capture levels and RESTORES them on exit, so a
barge_stress run is non-destructive -- it must not leave the mic dialled to the rig's
levels (the 2026-06-21 failure: a stale 13% source survived a run and crippled the
next live session)."""
from __future__ import annotations

import time
import types

import pytest

from tools.autotest import ota_setup
from tools.autotest.voice_loop import _engine_args


class _FakeAmixerPactl:
    """Records every amixer/pactl call and answers the `sget` / `get-source-volume`
    reads with the pre-run state, so gain_pinner can capture + restore it."""

    def __init__(self, cap_raw="33", boost_raw="0", src_pct="13"):
        self.cap_raw, self.boost_raw, self.src_pct = cap_raw, boost_raw, src_pct
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        out = ""
        if argv[0] == "amixer" and "sget" in argv:
            ctrl = argv[-1]
            if ctrl == "Capture":
                out = f"  Front Left: Capture {self.cap_raw} [52%] [7.50dB] [on]\n"
            else:  # Internal Mic Boost
                out = f"  Front Left: {self.boost_raw} [0%] [0.00dB]\n"
        elif argv[0] == "pactl" and len(argv) > 1 and argv[1] == "get-source-volume":
            out = f"Volume: front-left: 8740 / {self.src_pct}% / -52.50 dB\n"
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)


def _ssets(calls):
    return [c for c in calls if c[0] == "amixer" and "sset" in c]


def _source_sets(calls):
    return [c for c in calls if c[0] == "pactl" and len(c) > 1 and c[1] == "set-source-volume"]


def test_gain_pinner_restores_pre_run_levels_on_exit(monkeypatch):
    fake = _FakeAmixerPactl(cap_raw="40", boost_raw="2", src_pct="13")
    monkeypatch.setattr(ota_setup.subprocess, "run", fake)

    with ota_setup.gain_pinner(period_s=0.02):
        time.sleep(0.08)  # let the pin loop tick a few times

    # During the run it pinned the ADC down AND held the source UP.
    assert any(c[-2:] == ["Capture", f"{ota_setup.MIC_ADC_CAPTURE_PCT}%"] for c in _ssets(fake.calls))
    assert any(f"{ota_setup.MIC_SOURCE_VOLUME_PCT}%" in c for c in _source_sets(fake.calls))

    # On exit it RESTORED the captured pre-run raw levels (last write wins).
    last_cap = [c for c in _ssets(fake.calls) if "Capture" in c][-1]
    assert last_cap[-1] == "40"            # restored, not left at 52%
    last_boost = [c for c in _ssets(fake.calls) if "Internal Mic Boost" in c][-1]
    assert last_boost[-1] == "2"
    assert _source_sets(fake.calls)[-1][-1] == "13%"   # source restored to pre-run


def test_gain_pinner_is_safe_when_reads_return_nothing(monkeypatch):
    """If the level readers can't parse (e.g. no such control), restore is skipped
    rather than crashing -- the pinner must never break the harness."""
    def blank(argv, **kw):
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(ota_setup.subprocess, "run", blank)
    with ota_setup.gain_pinner(period_s=0.02):
        time.sleep(0.05)  # no exception


def test_autotest_engine_args_preserve_production_hybrid_models():
    args = _engine_args(
        "ollama",
        "gemma3:12b",
        "minicpm5-1b:q8",
        real_device=False,
    )

    assert args[args.index("--model") + 1] == "gemma3:12b"
    assert args[args.index("--fast-model") + 1] == "minicpm5-1b:q8"
