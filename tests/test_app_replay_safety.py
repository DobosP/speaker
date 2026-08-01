from __future__ import annotations

import pytest

from core.app import _run_replay


def test_generic_replay_refuses_synchronized_diagnostic_bundle(tmp_path) -> None:
    (tmp_path / "run-001.diagnostic.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run-001.wav").write_bytes(b"not-a-fixture")

    with pytest.raises(SystemExit, match="not a directory of replay utterances"):
        _run_replay(object(), object(), str(tmp_path))
