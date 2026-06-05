from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable

# Tier-0 selector: the whole tree MINUS every non-default tier and every
# dep/service-gated marker. This is the canonical CI-safe, logic-only set that
# the `unit` (and its alias `fast`) stage runs. Excludes: slow + e2e (Tier-1
# sim/subprocess), real_model + recorded (Tier-2 weights), live_output (Tier-3
# sound), and the cross-cutting network/llm/backend/hardware/discovery markers.
_NOT_TIERED = (
    "not slow and not e2e and not real_model and not live_output and not recorded "
    "and not network and not llm and not backend and not hardware and not discovery"
)


@dataclass(frozen=True)
class TestStage:
    name: str
    purpose: str
    paths: tuple[str, ...] = ("tests",)
    extra_args: tuple[str, ...] = ("-q",)
    timeout_sec: int | None = None
    allow_failures: bool = False
    # A command run BEFORE pytest; a non-zero exit aborts the stage (used by the
    # live tier to gate on `tools.live_session --check` -- models + audio ready).
    preflight: tuple[str, ...] | None = None
    # Extra environment for the pytest subprocess, as (key, value) pairs so the
    # frozen dataclass stays hashable. The live tier sets SPEAKER_LIVE=1 here.
    env: tuple[tuple[str, str], ...] = field(default=())

    def pytest_args(self, maxfail: int | None = None) -> list[str]:
        args = [*self.paths]
        if maxfail is not None:
            args.append(f"--maxfail={maxfail}")
        args.extend(self.extra_args)
        return args


@dataclass
class StageRegistry:
    stages: dict[str, TestStage]

    @classmethod
    def default(cls) -> "StageRegistry":
        stages = [
            TestStage(
                name="core",
                purpose="Runtime, action brain, and brain logic (fast, no models).",
                paths=(
                    "tests/test_core_runtime.py",
                    "tests/test_core_agent.py",
                    "tests/test_core_multimodal.py",
                    "tests/test_device_profiles.py",
                    "tests/test_specsim.py",
                    "tests/test_always_on_agent.py",
                    "tests/test_speaker_gate.py",
                ),
            ),
            TestStage(
                name="sandbox",
                purpose="Realistic-timing/concurrency middle-layer tests.",
                paths=("tests/test_sandbox_middle_layer.py",),
            ),
            TestStage(
                name="memory",
                purpose="Smart-memory save/writer logic + pool concurrency contract. "
                        "Integration tests run only with --pytest-arg=--postgres.",
                paths=(
                    "tests/test_memory_smart_save.py",
                    "tests/test_memory_writer.py",
                    "tests/test_memory_pool.py",
                    "tests/test_memory_postgres_integration.py",
                ),
            ),
            TestStage(
                name="cloud",
                purpose="Cloud LLM middle layer: providers, hedge chain, sensitivity routing, end-to-end integration.",
                paths=(
                    "tests/test_multi_provider_llm.py",
                    "tests/test_hedge_chain.py",
                    "tests/test_hedge_chain_advanced.py",
                    "tests/test_cloud_providers.py",
                    "tests/test_routing_intent.py",
                    "tests/test_sensitivity.py",
                    "tests/test_cloud_integration.py",
                    "tests/test_capability_context_isolation.py",
                ),
            ),
            TestStage(
                name="imports",
                purpose="Whole-tree import smoke: every first-party module compiles + libs resolve.",
                paths=("tests/test_imports_smoke.py",),
            ),
            TestStage(
                name="unit",
                purpose="Tier 0: pure logic + fakes -- no models, audio, subprocess, or "
                        "services. The CI-safe, logic-only set + the everyday TDD loop.",
                extra_args=("-q", "-m", _NOT_TIERED),
            ),
            TestStage(
                name="fast",
                purpose="Alias of `unit` -- the everyday TDD loop (Tier-0 logic only).",
                extra_args=("-q", "-m", _NOT_TIERED),
            ),
            TestStage(
                name="e2e",
                purpose="Tier 1: full end-to-end CLI/process tests (subprocess the real `python -m core`).",
                extra_args=("-q", "-m", "e2e"),
            ),
            TestStage(
                name="real_model",
                purpose="Tier 2: real trained weights over FIXTURES, no sound card (sherpa "
                        "ASR/TTS, two-pass final, Smart-Turn, replay). Self-skips when models absent.",
                extra_args=("-q", "-m", "real_model or recorded"),
            ),
            TestStage(
                name="live",
                purpose="Tier 3: REAL speakers/mic (live_session/real_usage). OPT-IN ONLY -- "
                        "preflights models+audio (tools.live_session --check), sets SPEAKER_LIVE=1, "
                        "then runs the live_output tests.",
                extra_args=("-q", "-m", "live_output"),
                preflight=(sys.executable, "-m", "tools.live_session", "--check"),
                env=(("SPEAKER_LIVE", "1"),),
            ),
            TestStage(
                name="full",
                purpose="The entire test suite.",
                paths=("tests",),
            ),
        ]
        return cls({stage.name: stage for stage in stages})

    def get(self, name: str) -> TestStage:
        try:
            return self.stages[name]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise ValueError(f"Unknown stage {name!r}. Known stages: {known}") from exc

    def names(self) -> list[str]:
        return list(self.stages)

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": stage.name,
                "purpose": stage.purpose,
                "paths": list(stage.paths),
                "allow_failures": stage.allow_failures,
            }
            for stage in self.stages.values()
        ]

    def select(self, names: Iterable[str]) -> list[TestStage]:
        return [self.get(name) for name in names]
