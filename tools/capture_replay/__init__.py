"""Deterministic, production-path microphone capture replay."""

from .corpus import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    LoadedReplayCorpus,
    ReplayAssertion,
    ReplayCase,
    ReplayCorpusError,
    ReplayTrack,
    SampleInterval,
    SpeakerInterval,
    SpeakerRole,
    WordInterval,
    load_corpus,
    verify_corpus_snapshot,
)
from .replay import ReplayCaptureSession

__all__ = [
    "FRAME_SAMPLES",
    "SAMPLE_RATE_HZ",
    "LoadedReplayCorpus",
    "ReplayAssertion",
    "ReplayCase",
    "ReplayCaptureSession",
    "ReplayCorpusError",
    "ReplayTrack",
    "SampleInterval",
    "SpeakerInterval",
    "SpeakerRole",
    "WordInterval",
    "load_corpus",
    "verify_corpus_snapshot",
]
