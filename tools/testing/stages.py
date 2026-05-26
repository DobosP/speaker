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
                purpose="Smart-memory save/writer logic.",
                paths=(
                    "tests/test_memory_smart_save.py",
                    "tests/test_memory_writer.py",
                ),
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
