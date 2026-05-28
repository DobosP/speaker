"""Tests for tools.recommend_profile: the hardware-probe decision tree and
the report formatter. Injects HostInfo directly so the tests don't depend
on the actual host running them."""
from __future__ import annotations

import json
import os

from tools.recommend_profile import (
    HostInfo,
    format_report,
    main,
    probe,
    recommend,
)


# --- decision tree ------------------------------------------------------

def test_recommend_nvidia_24gb_gets_desktop_gpu_4090():
    info = HostInfo(cores=16, ram_gb=64, gpu_kind="nvidia", gpu_mem_gb=24)
    profile, _ = recommend(info)
    assert profile == "desktop_gpu_4090"


def test_recommend_nvidia_12gb_falls_back_to_desktop():
    info = HostInfo(cores=12, ram_gb=32, gpu_kind="nvidia", gpu_mem_gb=12)
    profile, _ = recommend(info)
    assert profile == "desktop"


def test_recommend_apple_silicon_16gb_gets_macbook_profile():
    info = HostInfo(cores=10, ram_gb=16, gpu_kind="apple", gpu_mem_gb=16)
    profile, _ = recommend(info)
    assert profile == "macbook_m_series"


def test_recommend_cpu_only_with_16gb_gets_cpu_laptop():
    info = HostInfo(cores=8, ram_gb=16, gpu_kind=None, gpu_mem_gb=0)
    profile, rationale = recommend(info)
    assert profile == "cpu_laptop"
    assert "cloud" in rationale.lower()  # the rationale flags the cloud-hedge hint


def test_recommend_low_ram_desktop_uses_cpu_laptop_not_phone():
    """A 12 GB Linux laptop without GPU is still a laptop, not a phone --
    don't accidentally drop it into the GGUF/llamacpp mobile profile."""
    info = HostInfo(cores=8, ram_gb=12, gpu_kind=None, gpu_mem_gb=0, mobile=False)
    profile, _ = recommend(info)
    assert profile == "cpu_laptop"


def test_recommend_mobile_class_8gb_gets_phone_when_mobile_flag_set():
    info = HostInfo(cores=8, ram_gb=12, gpu_kind=None, gpu_mem_gb=0, mobile=True)
    profile, _ = recommend(info)
    assert profile == "phone"


def test_recommend_mobile_low_end_6gb_gets_phone_lite():
    info = HostInfo(cores=6, ram_gb=6, gpu_kind=None, gpu_mem_gb=0, mobile=True)
    profile, rationale = recommend(info)
    assert profile == "phone_lite"
    assert "cloud" in rationale.lower()


def test_recommend_desktop_under_8gb_falls_to_lite_with_cloud_hint():
    info = HostInfo(cores=4, ram_gb=6, gpu_kind=None, gpu_mem_gb=0, mobile=False)
    profile, rationale = recommend(info)
    assert profile == "phone_lite"
    assert "cloud" in rationale.lower()


def test_recommend_handles_undetected_ram_with_lite_fallback():
    info = HostInfo(cores=2, ram_gb=0.0, gpu_kind=None, gpu_mem_gb=0)
    profile, _ = recommend(info)
    assert profile == "phone_lite"  # fail-safe, never raises


# --- probe injection ----------------------------------------------------

def test_probe_prefers_injected_values_over_real_host():
    info = probe(ram_gb=64.0, nvidia_mem_gb=24.0, apple=False, cores=16)
    assert info.cores == 16
    assert info.ram_gb == 64.0
    assert info.gpu_kind == "nvidia"
    assert info.gpu_mem_gb == 24.0


def test_probe_apple_branch_uses_ram_as_gpu_mem():
    info = probe(ram_gb=32.0, nvidia_mem_gb=0.0, apple=True, cores=10)
    assert info.gpu_kind == "apple"
    assert info.gpu_mem_gb == 32.0


def test_probe_falls_through_to_cpu_only_when_no_gpu():
    info = probe(ram_gb=16.0, nvidia_mem_gb=0.0, apple=False, cores=8)
    assert info.gpu_kind is None
    assert info.gpu_mem_gb == 0.0


# --- report formatter ---------------------------------------------------

def test_format_report_includes_probe_recommendation_and_run_hint():
    info = HostInfo(cores=8, ram_gb=16, gpu_kind=None, gpu_mem_gb=0)
    out = format_report(info, "cpu_laptop", "no gpu, cloud recommended")
    assert "8" in out
    assert "16.0 GB" in out
    assert "cpu_laptop" in out
    assert "python -m core --device cpu_laptop" in out


def test_format_report_describes_nvidia_and_apple_gpu_kinds():
    nvidia = HostInfo(cores=16, ram_gb=64, gpu_kind="nvidia", gpu_mem_gb=24)
    apple = HostInfo(cores=10, ram_gb=32, gpu_kind="apple", gpu_mem_gb=32)
    assert "NVIDIA" in format_report(nvidia, "desktop_gpu_4090", "")
    assert "Apple Silicon" in format_report(apple, "macbook_m_series", "")


# --- end-to-end main() smoke (probes the real host, just asserts no crash)

def test_main_smoke(capsys):
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Recommended profile" in captured.out
    assert "python -m core --device" in captured.out


# --- profile-name consistency with config.json --------------------------

def test_every_recommended_profile_exists_in_config_json():
    """If recommend() returns a name, the matching device_profile must
    actually be defined in config.json -- otherwise the user's first
    ``--device <name>`` run would silently fall back to the base config."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    profiles = set((cfg.get("device_profiles") or {}).keys())
    # Walk every decision-tree branch.
    for info in [
        HostInfo(16, 64, "nvidia", 24, mobile=False),
        HostInfo(12, 32, "nvidia", 12, mobile=False),
        HostInfo(10, 16, "apple", 16, mobile=False),
        HostInfo(8, 16, None, 0, mobile=False),
        HostInfo(8, 12, None, 0, mobile=False),  # cpu_laptop (desktop OS)
        HostInfo(8, 12, None, 0, mobile=True),   # phone (mobile OS)
        HostInfo(6, 6, None, 0, mobile=True),
        HostInfo(4, 6, None, 0, mobile=False),   # phone_lite fallback on desktop
        HostInfo(2, 0, None, 0),
    ]:
        profile, _ = recommend(info)
        assert profile in profiles, (
            f"recommend() returned {profile!r} but config.json device_profiles "
            f"only has {sorted(profiles)}"
        )
