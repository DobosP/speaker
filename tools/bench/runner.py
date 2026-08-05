from __future__ import annotations

import glob
import gc
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from always_on_agent.events import Mode

from core.metrics import TTS_FIRST_AUDIO, TurnRecord
from core.runtime import VoiceRuntime


@dataclass
class TurnSample:
    """One benchmarked turn: the latency record plus correctness context."""

    name: str
    expectation: str
    transcript: str
    responded: bool
    record: TurnRecord
    reference: str = field(default="", repr=False)
    assertion: str = "legacy"
    commands: tuple[str, ...] = field(default=(), repr=False)
    forbidden_commands: tuple[str, ...] = field(default=(), repr=False)
    replay_decode_seconds: float | None = None
    replay_to_idle_seconds: float | None = None

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


class ProductionBenchError(RuntimeError):
    """A detail-free production replay failure."""


@dataclass(frozen=True)
class ProductionRun:
    """Private samples plus safe identities for one production-profile replay."""

    samples: tuple[TurnSample, ...] = field(repr=False)
    device_profile: str
    final_stt_profile: str
    final_stt_profile_sha256: str
    final_stt_profile_schema_version: int
    production_config_sha256: str
    execution_config_sha256: str
    replay_safety_schema_version: int
    replay_safety_sha256: str
    active_artifacts_sha256: str
    active_artifact_roots: int
    active_artifact_files: int
    active_artifact_bytes: int
    source_revision: str
    runtime_identities: dict[str, object]
    ollama_identity_sha256: str
    ollama_role_contracts: dict[str, dict[str, str]]


@contextmanager
def _production_replay_log_guard():
    """Temporarily suppress logging while private corpus text is in flight.

    Production runtime INFO messages can interpolate raw ASR finals, cleaned
    text, prompts, or spoken replies.  The process-wide logging threshold also
    covers last-resort handlers and loggers/handlers created during the row;
    the prior threshold is restored on every exit path.  This evaluator owns
    the process while a row runs, so losing diagnostic logs is preferable to
    leaking retained speech into an inherited logging configuration.
    """
    previous_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_threshold)


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


def run_production(
    corpus,
    config: dict,
    *,
    device_profile: str,
    final_stt_profile: str,
    final_stt_profile_sha256: str,
    final_stt_profile_schema_version: int,
    production_config_sha256: str,
    execution_config_sha256: str,
    replay_safety_schema_version: int,
    replay_safety_sha256: str,
    active_artifacts_sha256: str,
    active_artifact_roots: int,
    active_artifact_files: int,
    active_artifact_bytes: int,
    source_revision: str,
    runtime_identities: dict[str, object],
    ollama_identity_sha256: str,
    ollama_role_contracts: dict[str, dict[str, str]],
    timeout: float = 120.0,
    after_build=None,
) -> ProductionRun:
    """Run one strict corpus sequentially through the production assembly.

    The transport is deliberately FileReplay: complete corpus PCM is available
    before each case, TTS is synthesized to a null sink, and no audio device is
    opened.  The caller owns configuration/profile/corpus validation and repeats
    their source bindings after this function returns.
    """
    import numpy as np

    from core.app import build_runtime
    from core.engines.file_replay import FileReplayEngine
    from core.engines.sherpa import SherpaConfig
    from core.llm import OllamaLLM
    from core.llm_factory import build_llms
    from core.routing import build_router
    from tools.conversation_eval.provenance import LOCAL_OLLAMA_HEADERS

    sherpa_cfg = SherpaConfig.from_dict(config.get("sherpa", {}))
    # FileReplay otherwise permits a missing TTS and merely reports no answer.
    # Production whole-turn evidence requires the configured recognizer and TTS.
    required = (
        "asr_encoder",
        "asr_tokens",
        "vad_model",
        "tts_model",
        "tts_tokens",
    )
    if any(not str(getattr(sherpa_cfg, key, "") or "").strip() for key in required):
        raise ProductionBenchError()

    llm_backend = str((config.get("llm", {}) or {}).get("backend", "ollama"))
    llm_args = SimpleNamespace(
        llm=llm_backend,
        model=None,
        fast_model=None,
        ollama_client_headers=dict(LOCAL_OLLAMA_HEADERS),
    )
    main_llm, fast_llm = build_llms(llm_args, config)
    # The production-input boundary admits one local Ollama main/fast pair.  Do
    # not let a cloud wrapper, echo stub, missing fast tier, or a factory that
    # ignored the identity probe's explicit local headers enter this evidence
    # path merely because it implements the broad LLM protocol.
    expected_headers = dict(LOCAL_OLLAMA_HEADERS)
    if (
        type(main_llm) is not OllamaLLM
        or type(fast_llm) is not OllamaLLM
        or getattr(main_llm, "_client_headers", None) != expected_headers
        or getattr(fast_llm, "_client_headers", None) != expected_headers
    ):
        raise ProductionBenchError()

    samples: list[TurnSample] = []
    for case_index, case in enumerate(corpus.cases):
        # Each row owns a fresh mutable runtime: metrics, recent conversation,
        # memory, supervisor/controller state, router observations, and replay
        # engine cannot flow into the next corpus assertion.  The stateless
        # Ollama provider clients remain shared so every row uses the exact same
        # production-built transport/model identity and server residency.
        engine = FileReplayEngine(sherpa_cfg)
        runtime = build_runtime(
            config,
            engine=engine,
            llm=main_llm,
            fast_llm=fast_llm,
            router=build_router(config),
            start_mode=Mode.ASSISTANT,
        )
        with _production_replay_log_guard():
            started = False
            try:
                runtime.start(run_bus=False)
                started = True
                warm_ready = getattr(runtime, "warm_ready", None)
                wait_warm = getattr(warm_ready, "wait", None)
                if not callable(wait_warm) or not wait_warm(timeout=timeout):
                    raise ProductionBenchError()
                if case_index == 0 and after_build is not None:
                    # Engine/model construction and configured best-effort LLM
                    # warm-up are now complete, so this boundary resource sample
                    # actually observes the resident production assembly.
                    after_build()

                # Warm-up is complete before either baseline accounting or the
                # complete-PCM replay timer begins.
                runtime.metrics.close_turn()
                records_before = len(runtime.metrics.records())
                finals_before = len(engine.finals)
                engine.last_final = ""

                pcm = np.frombuffer(case.audio_bytes, dtype="<f4")
                case_started = time.perf_counter()
                engine.replay_samples(pcm, 16_000)
                decode_finished = time.perf_counter()
                if not runtime.wait_idle(timeout=timeout):
                    raise ProductionBenchError()
                idle_finished = time.perf_counter()
                runtime.metrics.close_turn()

                new_records = runtime.metrics.records()[records_before:]
                new_finals = engine.finals[finals_before:]
                if len(new_records) > 1 or len(new_finals) > 1:
                    # A strict corpus row represents exactly one whole turn.  Do
                    # not silently align a multi-final replay to one reference.
                    raise ProductionBenchError()
                record = new_records[0] if new_records else TurnRecord()
                transcript = new_finals[0] if new_finals else ""
                samples.append(
                    TurnSample(
                        name=case.case_id,
                        expectation=case.assertion,
                        transcript=transcript,
                        # FileReplay appends to ``spoken`` before synthesis.
                        # Only its completed-clip TTS metric proves that an
                        # answer made it through the configured TTS boundary.
                        responded=TTS_FIRST_AUDIO in record.stamps,
                        record=record,
                        reference=case.expected_text,
                        assertion=case.assertion,
                        commands=case.commands,
                        forbidden_commands=case.forbidden_commands,
                        replay_decode_seconds=decode_finished - case_started,
                        replay_to_idle_seconds=idle_finished - case_started,
                    )
                )
            finally:
                if started:
                    runtime.stop()
        # FileReplay callbacks retain bound runtime methods; explicitly drop
        # the per-row cycle so repeated native ASR/TTS construction cannot
        # accumulate model memory across a larger corpus.
        del runtime
        del engine
        gc.collect()

    return ProductionRun(
        samples=tuple(samples),
        device_profile=device_profile,
        final_stt_profile=final_stt_profile,
        final_stt_profile_sha256=final_stt_profile_sha256,
        final_stt_profile_schema_version=final_stt_profile_schema_version,
        production_config_sha256=production_config_sha256,
        execution_config_sha256=execution_config_sha256,
        replay_safety_schema_version=replay_safety_schema_version,
        replay_safety_sha256=replay_safety_sha256,
        active_artifacts_sha256=active_artifacts_sha256,
        active_artifact_roots=active_artifact_roots,
        active_artifact_files=active_artifact_files,
        active_artifact_bytes=active_artifact_bytes,
        source_revision=source_revision,
        runtime_identities=runtime_identities,
        ollama_identity_sha256=ollama_identity_sha256,
        ollama_role_contracts=ollama_role_contracts,
    )
