from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TestStage:
    name: str
    purpose: str
    paths: tuple[str, ...] = ("tests",)
    markers: str | None = None
    extra_args: tuple[str, ...] = ()
    timeout_sec: int | None = None
    allow_failures: bool = False

    def pytest_args(self, maxfail: int | None = None) -> list[str]:
        args = [*self.paths]
        if self.markers:
            args.extend(["-m", self.markers])
        if maxfail is not None:
            args.append(f"--maxfail={maxfail}")
        args.extend(self.extra_args)
        return args


@dataclass
class StageRegistry:
    stages: dict[str, TestStage] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "StageRegistry":
        stages = [
            TestStage(
                name="smoke",
                purpose="Fast import/config/schema checks.",
                paths=("tests/test_profiles.py", "tests/test_test_runner_quality.py"),
                markers="smoke",
                extra_args=("-q",),
            ),
            TestStage(
                name="dev",
                purpose="Critical fast tests for everyday TDD.",
                paths=(
                    "tests/test_conversation_simulation.py",
                    "tests/test_profiles.py",
                    "tests/test_test_runner_quality.py",
                ),
                markers="dev",
                extra_args=("-q",),
            ),
            TestStage(
                name="audio",
                purpose="Deterministic audio, VAD, barge-in, replay, and conversation tests.",
                paths=(
                    "tests/test_conversation_simulation.py",
                    "tests/test_bargein_scenarios.py",
                    "tests/test_recorded_sessions.py",
                ),
                markers="audio and not discovery",
                extra_args=("-q",),
            ),
            TestStage(
                name="replay",
                purpose="Recorded-session replay and recording metadata checks.",
                paths=("tests/test_recorded_sessions.py",),
                markers="recorded",
                extra_args=("-q",),
            ),
            TestStage(
                name="discovery",
                purpose="Failure-discovery corpus; expected to fail until bugs are fixed.",
                paths=(
                    "tests/test_failure_discovery_audio.py",
                    "tests/test_acoustic_realism.py",
                    "tests/test_noise_robustness.py",
                    "tests/test_voice_variability.py",
                ),
                markers="discovery",
                extra_args=("-q",),
                allow_failures=True,  # scripts/run_tests.py: non-zero pytest RC still exits 0 (allowed_to_fail)
            ),
            TestStage(
                name="backend",
                purpose="Optional STT/TTS/LLM backend tests.",
                paths=(
                    "tests/test_stt_backends.py",
                    "tests/test_tts_backends.py",
                    "tests/test_llm.py",
                ),
                markers="backend",
                extra_args=("-q",),
                allow_failures=True,
            ),
            TestStage(
                name="full",
                purpose="All normal tests except discovery/backend/hardware/network/LLM gates.",
                paths=("tests",),
                markers="not discovery and not backend and not hardware and not network and not llm",
                extra_args=("-q",),
            ),
            TestStage(
                name="all",
                purpose="Everything, including discovery and optional backend tests.",
                paths=("tests",),
                extra_args=("-q",),
                allow_failures=True,
            ),
        ]
        return cls({stage.name: stage for stage in stages})

    def get(self, name: str) -> TestStage:
        try:
            return self.stages[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.stages))
            raise ValueError(f"Unknown stage {name!r}. Known stages: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self.stages)

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": stage.name,
                "purpose": stage.purpose,
                "paths": list(stage.paths),
                "markers": stage.markers,
                "allow_failures": stage.allow_failures,
            }
            for stage in self.stages.values()
        ]

    def select(self, names: Iterable[str]) -> list[TestStage]:
        return [self.get(name) for name in names]
