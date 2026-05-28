from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TestStage:
    name: str
    purpose: str
    paths: tuple[str, ...] = ("tests",)
    extra_args: tuple[str, ...] = ("-q",)
    timeout_sec: int | None = None
    allow_failures: bool = False

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
