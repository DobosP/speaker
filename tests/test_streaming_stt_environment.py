from __future__ import annotations

import importlib
from pathlib import Path
import sys

from tools.install import RUNTIME_DEPS
from tools.streaming_stt.supervisor import sanitized_worker_environment


def test_controller_import_never_imports_candidate_model_runtimes():
    before = set(sys.modules)
    importlib.import_module("tools.streaming_stt_eval")
    newly_loaded = set(sys.modules) - before

    assert not any(
        name == candidate or name.startswith(candidate + ".")
        for candidate in ("torch", "transformers", "moonshine", "moonshine_voice")
        for name in newly_loaded
    )


def test_worker_module_import_is_inert_without_launch_bootstrap():
    worker = importlib.import_module("tools.streaming_stt.worker")

    assert worker._SOURCE_BUNDLE_FD is None
    assert worker._SOURCE_BUNDLE_SHA256 is None


def test_worker_environment_is_secret_free_offline_and_single_threaded(tmp_path):
    environment = sanitized_worker_environment(Path(tmp_path))

    assert "PYTHONPATH" not in environment
    assert "HF_TOKEN" not in environment
    assert "HOME" in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["MOONSHINE_ORT_SINGLE_THREAD"] == "1"
    assert {
        environment["OMP_NUM_THREADS"],
        environment["OPENBLAS_NUM_THREADS"],
        environment["MKL_NUM_THREADS"],
        environment["NUMEXPR_NUM_THREADS"],
    } == {"1"}


def test_harness_adds_no_candidate_dependency_to_lean_runtime():
    joined = "\n".join(RUNTIME_DEPS).lower()

    assert "torch" not in joined
    assert "transformers" not in joined
    assert "moonshine" not in joined
