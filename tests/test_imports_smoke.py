"""Whole-tree import smoke test.

Walks every importable Python module under the first-party packages (core,
always_on_agent, remote, tools, utils) and confirms each one imports cleanly
under the current interpreter + installed dependencies. Catches three classes
of regression that pure unit tests miss:

- **Syntax errors** in modules that aren't covered by any test (Python
  doesn't compile ahead-of-time; an import is the equivalent check).
- **Missing optional dependencies** the codebase advertises in its setup --
  e.g. running tests without ``ollama``/``openai`` installed used to
  silently skip whole code paths.
- **Stale imports** (a renamed symbol that some peripheral module still
  references) -- those only surface when the module is loaded.

Heavy optional modules (engines that need sherpa-onnx ML weights, LiveKit
transport, GPU torch) are listed in ``_OPTIONAL`` and skipped with an
xfail-style note. Add to that list when a new module needs a heavy dep
that the CI environment doesn't have.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Iterator

import pytest

# Packages whose modules we want imported. Tests/ and tools/testing/ live
# outside because they're pytest-style and would re-trigger the runner.
_PACKAGES = ("core", "always_on_agent", "remote", "tools", "utils")

# Modules that legitimately need heavy/optional deps not installed in the
# minimal test environment. If you skip something here, prefer to also
# document WHY (so the next reader doesn't widen the list lazily).
_OPTIONAL = {
    # sherpa-onnx engines: need the C++ extension + downloaded ONNX models.
    "core.engines.sherpa": "needs sherpa-onnx + downloaded ASR/TTS models",
    "core.engines.file_replay": "needs sherpa-onnx for the recognizer/TTS",
    "core.engines.livekit": "needs livekit + livekit-api wheels",
    "core.engines.speaker_gate": "needs sherpa-onnx for the embedding model",
    # Remote worker drags in livekit-api.
    "remote.worker": "needs livekit + livekit-api wheels",
    # Open Interpreter brain.
    "core.agent": "needs open-interpreter",
    # CUDA / GPU bits.
    "core.engines.audio": "needs sounddevice + sherpa-onnx",
    # Tools that pull in heavy deps when imported.
    "tools.bench": "needs sherpa-onnx + huggingface_hub for model fetch",
    "tools.cloudchat": "needs openai SDK at import time",
    # Smart-memory module (Postgres + pgvector + embeddings).
    "utils.memory": "needs numpy + psycopg + embeddings",
    "utils.memory_writer": "needs numpy + psycopg",
    "utils.memory_config": "needs psycopg",
}


def _iter_first_party_modules() -> Iterator[str]:
    """Yield every importable module under ``_PACKAGES``."""
    repo_root = Path(__file__).resolve().parent.parent
    for pkg_name in _PACKAGES:
        pkg_path = repo_root / pkg_name
        if not pkg_path.is_dir():
            continue
        # Yield the package itself.
        if (pkg_path / "__init__.py").exists():
            yield pkg_name
        for _, mod_name, _is_pkg in pkgutil.walk_packages([str(pkg_path)], prefix=f"{pkg_name}."):
            if mod_name.startswith(f"{pkg_name}.__"):
                continue
            yield mod_name


@pytest.mark.parametrize("module_name", sorted(set(_iter_first_party_modules())))
def test_module_imports(module_name):
    """Each first-party module must import without raising.

    Modules with known-heavy optional deps are documented in ``_OPTIONAL``
    and skipped (the import is still attempted -- a missing dep registers
    as a skip, NOT a failure). Any *other* ImportError or SyntaxError fails
    the test."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if module_name in _OPTIONAL:
            pytest.skip(f"{module_name}: {_OPTIONAL[module_name]} ({exc.name})")
        raise
    except ImportError as exc:
        if module_name in _OPTIONAL:
            pytest.skip(f"{module_name}: {_OPTIONAL[module_name]} ({exc})")
        raise


def test_core_public_surface_is_importable():
    """A focused canary: the public-API touchpoints we lean on heavily."""
    targets = [
        ("core.app", "_wrap_cloud"),
        ("core.app", "_build_cloud_client"),
        ("core.llm", "HedgeLLM"),
        ("core.llm", "SensitivityRouterLLM"),
        ("core.llm", "capability_context"),
        ("core.llm", "OpenAICompatLLM"),
        ("core.routing", "HeuristicRouter"),
        ("core.routing", "ChainSelector"),
        ("core.routing", "build_chain_selector"),
        ("core.sensitivity", "classify_sensitivity"),
        ("core.sensitivity", "PRIVATE"),
        ("core.sensitivity", "CODE"),
        ("core.sensitivity", "PUBLIC"),
        ("core.capabilities", "attach_llm_capabilities"),
        ("always_on_agent.models", "IntentKind"),
        ("always_on_agent.events", "Mode"),
    ]
    for module, attr in targets:
        mod = importlib.import_module(module)
        assert hasattr(mod, attr), f"{module}.{attr} missing -- API contract regressed"


def test_first_party_package_count_is_above_floor():
    """Sanity floor: if a refactor accidentally renames a top-level package,
    this catches the disappearance instead of the suite silently passing
    with zero modules to walk."""
    n = sum(1 for _ in _iter_first_party_modules())
    assert n > 25, (
        f"first-party module enumeration only found {n} modules -- "
        f"check that {_PACKAGES} still exist and are Python packages"
    )
