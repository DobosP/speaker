from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from always_on_agent.events import Mode

from core.metrics import TurnRecord
from core.runtime import VoiceRuntime


@dataclass
class TurnSample:
    """One benchmarked turn: the latency record plus correctness context."""

    name: str
    expectation: str
    transcript: str
    responded: bool
    record: TurnRecord

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expectation": self.expectation,
            "transcript": self.transcript,
            "responded": self.responded,
            **self.record.as_dict(),
        }


@dataclass
class Fixture:
    name: str
    path: str
    expectation: str = ""


def discover_fixtures(directory: str, limit: Optional[int] = None) -> list[Fixture]:
    """List ``.npy``/``.wav`` fixtures in ``directory``, attaching the
    ``expectation`` from a sibling ``metadata.json`` when present."""
    meta: dict[str, str] = {}
    meta_path = os.path.join(directory, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for case in data.get("cases", []):
            if isinstance(case, dict) and case.get("name"):
                meta[str(case["name"])] = str(case.get("expectation", ""))

    paths = sorted(glob.glob(os.path.join(directory, "*.npy")))
    paths += sorted(glob.glob(os.path.join(directory, "*.wav")))
    fixtures = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        fixtures.append(Fixture(name=stem, path=path, expectation=meta.get(stem, "")))
    if limit is not None:
        fixtures = fixtures[:limit]
    return fixtures


def _collect_aligned(runtime: VoiceRuntime, count: int) -> list[TurnRecord]:
    """Return exactly ``count`` records (pad with empty ones if a case produced
    no turn, e.g. an utterance the brain ignored in passive mode)."""
    runtime.metrics.close_turn()
    records = runtime.metrics.records()
    if len(records) < count:
        records = records + [TurnRecord() for _ in range(count - len(records))]
    return records[:count]


def run_fake(cases: list[tuple[str, str]], *, timeout: float = 5.0) -> list[TurnSample]:
    """Dependency-free smoke run: ScriptedEngine + EchoLLM drive each case so the
    metrics + report plumbing can be validated without models or downloads.

    ``cases`` is a list of ``(name, utterance)``. Latencies will be near-zero
    (everything is instant) -- the point is that records populate and align."""
    from core.engines.scripted import ScriptedEngine
    from core.llm import EchoLLM

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    # Pump the bus synchronously (run_bus=False): wait_idle drains pending
    # TTS_REQUESTs in-thread so tts_first_audio is stamped into the right turn
    # before we close it -- an async bus races the turn boundary.
    runtime.start(run_bus=False)
    spoke_before = 0
    responded: list[bool] = []
    try:
        for _name, utterance in cases:
            runtime.metrics.close_turn()
            engine.final(utterance)
            runtime.wait_idle(timeout=timeout)
            responded.append(len(engine.spoken) > spoke_before)
            spoke_before = len(engine.spoken)
        records = _collect_aligned(runtime, len(cases))
    finally:
        runtime.stop()

    return [
        TurnSample(name, "callback", utterance, responded[i], records[i])
        for i, (name, utterance) in enumerate(cases)
    ]


def run_real(
    fixtures: list[Fixture],
    sherpa_cfg,
    main_llm,
    fast_llm=None,
    *,
    start_mode: Mode = Mode.ASSISTANT,
    stream_tts: bool = False,
    timeout: float = 60.0,
) -> list[TurnSample]:
    """Run the real pipeline over ``fixtures``. Requires sherpa-onnx + a model-
    backed LLM and configured model files in ``sherpa_cfg``."""
    from core.engines.file_replay import FileReplayEngine, load_waveform

    engine = FileReplayEngine(sherpa_cfg)
    runtime = VoiceRuntime(
        engine, main_llm, fast_llm=fast_llm, start_mode=start_mode, stream_tts=stream_tts
    )
    # Synchronous bus pump (see run_fake): keeps tts_first_audio in-turn so the
    # per-turn latency stamps are attributed correctly instead of racing.
    runtime.start(run_bus=False)
    transcripts: list[str] = []
    responded: list[bool] = []
    spoke_before = 0
    try:
        for fx in fixtures:
            runtime.metrics.close_turn()
            samples, sample_rate = load_waveform(fx.path)
            engine.last_final = ""
            engine.replay_samples(samples, sample_rate)
            runtime.wait_idle(timeout=timeout)
            transcripts.append(engine.last_final)
            responded.append(len(engine.spoken) > spoke_before)
            spoke_before = len(engine.spoken)
        records = _collect_aligned(runtime, len(fixtures))
    finally:
        runtime.stop()

    return [
        TurnSample(fx.name, fx.expectation, transcripts[i], responded[i], records[i])
        for i, fx in enumerate(fixtures)
    ]
