"""DTD coherence echo-veto regression tests.

The DTD path weights coherence at 0.0 in production, so a residual/raw z-score
spike can fire even when the reference-coherence detector has just classified
the same frame as echo-only. These tests drive the real Sherpa gate and real DTD
with a fake coherence verdict so the veto contract is pinned without audio
hardware or models.
"""
from __future__ import annotations

import pytest

from tests import barge_fixtures as bf


class _VerdictCoherence:
    def __init__(self, verdict, *, incoherent_fraction: float = 0.8) -> None:
        self._verdict = verdict
        self.last_incoherent_fraction = float(incoherent_fraction)

    def decide(self, mic_raw):
        return self._verdict


def _engine_with_warm_dtd(verdict, *, veto_enabled: bool = True):
    eng = bf.live_engine_with_dtd()
    eng.config.dtd_coherence_echo_veto = bool(veto_enabled)
    eng._echo_coherence = _VerdictCoherence(verdict)
    eng._dtd.new_run()

    raw_floor = 0.010
    resid_floor = 0.005
    for _ in range(8):
        eng._dtd.observe_echo(raw_floor, resid_floor, 0.8)
        eng._update_raw_playback_floor(raw_floor)
        eng._update_playback_floor(resid_floor)
    return eng


def _force_dtd_fire(eng) -> bool:
    result = eng._looks_like_user(bf.make_block(0.080), bf.make_block(0.080))
    assert eng._dtd.last_decided is True, "test setup did not produce a DTD fire"
    return result


def test_dtd_fire_is_vetoed_when_coherence_says_echo_only():
    eng = _engine_with_warm_dtd(False)

    assert _force_dtd_fire(eng) is False


@pytest.mark.parametrize("verdict", [True, None])
def test_dtd_fire_still_counts_when_coherence_confirms_or_abstains(verdict):
    eng = _engine_with_warm_dtd(verdict)

    assert _force_dtd_fire(eng) is True


def test_dtd_coherence_echo_veto_is_off_switchable():
    eng = _engine_with_warm_dtd(False, veto_enabled=False)

    assert _force_dtd_fire(eng) is True
