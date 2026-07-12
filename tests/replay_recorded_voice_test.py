"""Recorded-voice replay regression: the OWNER's real speech, replayed headless
through the REAL pipeline, asserting capabilities deterministically.

This is the assertion layer of the recorded-voice replay harness. It owns *zero*
pipeline wiring -- every seam (the sherpa-config skip idiom, FileReplay turn-taking,
the inject/barge-overlap path, metrics access) lives in ``tests/replay_voice_driver``;
the clip coordinates live in ``tests/fixtures/recorded_voice_manifest.json``. This
file only loads those two and asserts.

What it pins (CORPUS-grounded, latency is secondary):
  (1) Each clean owner utterance replays through the real ASR and the turn
      actually runs (ASR_FINAL stamped), with a lenient text match against the
      manifest's ``expected_text`` (substring OR >= 0.4 word overlap -- the
      ``test_asr_live_quality.py`` threshold -- so documented ASR variants like
      "listeningistening to me." don't flake it). EchoLLM makes the response
      deterministic, no network.
  (2) An ordered multi-clip sequence from ONE recorded session runs as a real
      multi-turn conversation through the real brain in a single runtime; each
      turn produces a final and the EchoLLM response echoes that turn's final.
  (3) Two explicit, hash-pinned same-session owner talk-overs each cut the
      assistant after a floor-only no-cut control, with consumed-sample and FIFO
      timestamps bound to the target metrics turn.

Tier 2 (``real_model``/``recorded``): needs the sherpa ASR models AND the locally
extracted owner clips, so it SELF-SKIPS on the logic-only CI and on any clean
clone without ``tools.extract_voice_clips`` having been run. It is exercised
post-merge by ``perf.yml`` (which selects ``-m "real_model or recorded"``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import tests.replay_voice_driver as replay_voice_driver

pytestmark = [pytest.mark.recorded, pytest.mark.real_model, pytest.mark.slow]

# The single source of truth for clip coordinates. Text/JSON only -- never voice
# audio (the extracted clips live in a gitignored local dir). This file imports
# ONLY the manifest for coordinates; it never inlines timestamps.
_MANIFEST_PATH = Path(__file__).parent / "fixtures" / "recorded_voice_manifest.json"

# The lenient ASR-match threshold reused verbatim from test_asr_live_quality.py:97
# -- the offline final refines streaming partials, it does not have to be a
# byte-for-byte transcript of the owner's (often low-RMS, real-room) audio.
_WORD_OVERLAP_FLOOR = 0.4


def _load_manifest() -> dict:
    """Parse the committed manifest, or an empty skeleton if it isn't present.

    Degrading to empty (rather than raising at import time) keeps collection
    green on a clean clone where the manifest hasn't landed yet: the tests below
    self-skip on an empty corpus instead of erroring at Tier 0. The real corpus
    is committed text/JSON, so on a normal checkout this returns it intact.
    """
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {"clips": [], "barge": [], "barge_cases": []}
    data.setdefault("clips", [])
    data.setdefault("barge", [])
    data.setdefault("barge_cases", [])
    return data


_MANIFEST = _load_manifest()
_CLIPS = [c for c in _MANIFEST["clips"] if isinstance(c, dict) and c.get("id")]
_BARGES = [b for b in _MANIFEST["barge"] if isinstance(b, dict) and b.get("id")]
_BARGE_CASES = [
    case
    for case in _MANIFEST.get("barge_cases", [])
    if isinstance(case, dict) and case.get("id")
]
_CLIP_BY_ID = {c["id"]: c for c in _CLIPS}
_BARGE_BY_ID = {b["id"]: b for b in _BARGES}


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- so the comparison is
    about *words*, not casing/period placement the recognizer may differ on."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def _matches(actual: str, expected: str) -> tuple[bool, float]:
    """Lenient match used everywhere a real-voice transcript is compared.

    Returns ``(ok, overlap)``. ``ok`` is true when either side contains the
    other as a substring (after normalization) OR the word overlap (intersection
    over the expected word count) reaches ``_WORD_OVERLAP_FLOOR``. The overlap is
    returned too so failures print a diagnostic number.
    """
    a = _normalize(actual)
    e = _normalize(expected)
    if not e:
        return (bool(a), 1.0 if a else 0.0)
    if e in a or (a and a in e):
        return (True, 1.0)
    e_words = e.split()
    a_words = set(a.split())
    if not e_words:
        return (False, 0.0)
    overlap = sum(1 for w in e_words if w in a_words) / len(e_words)
    return (overlap >= _WORD_OVERLAP_FLOOR, overlap)


# ---- (1) every clean owner utterance reproduces through the real ASR ----------

@pytest.mark.skipif(not _CLIPS, reason="recorded_voice_manifest.json has no clips")
@pytest.mark.parametrize("clip_id", [c["id"] for c in _CLIPS], ids=[c["id"] for c in _CLIPS])
def test_each_utterance_reproduced(monkeypatch, clip_id):
    """Replay one owner utterance headless through the REAL sherpa recognizer and
    EchoLLM; the turn must actually run (ASR_FINAL stamped) and the final must
    leniently match the manifest's expected_text. Deterministic, no network."""
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")
    scfg = replay_voice_driver.sherpa_config_or_skip()

    expected = _CLIP_BY_ID[clip_id].get("expected_text", "")
    results = replay_voice_driver.run_turns(scfg, [clip_id])
    assert results, f"driver produced no TurnResult for {clip_id!r}"
    turn = results[-1]

    # The turn actually ran: the metrics record stamped ASR_FINAL. The driver
    # surfaces this as TurnResult.asr_final_stamped (read off the TurnRecord's
    # stamps). An unstamped turn means the endpointer never fired -- a real
    # failure, not a skip.
    assert turn.asr_final_stamped, (
        f"{clip_id!r}: ASR_FINAL was never stamped -- the turn never reached "
        f"endpoint (expected ~{expected!r}, got final {turn.asr_final!r})"
    )

    # Latency is secondary (CORPUS): record it, never assert a bound.
    print(
        f"[{clip_id}] expected={expected!r} asr_final={turn.asr_final!r} "
        f"first_audio_latency={turn.first_audio_latency}"
    )

    ok, overlap = _matches(turn.asr_final, expected)
    assert ok, (
        f"{clip_id!r}: ASR final diverged from expected (word overlap "
        f"{overlap:.2f} < {_WORD_OVERLAP_FLOOR}). "
        f"expected={expected!r} got={turn.asr_final!r}"
    )


# ---- (2) a real multi-turn conversation from one recorded session --------------

# An ordered sequence captured in ONE live session (run-20260602-235431: "Tell me
# a long story" -> "Stop speaking"), driven back-to-back through a single runtime
# so it exercises the real brain's turn bookkeeping, not just isolated utterances.
# Only ids actually present in the committed manifest are used (so the manifest can
# list more than is extractable locally without breaking this test).
_MULTITURN_PREFERRED = ["utterance-01", "utterance-02"]


def _multiturn_clip_ids() -> list[str]:
    """Prefer the documented run-20260602-235431 sequence; otherwise fall back to
    the first clips that share a single ``run`` -- so this stays a genuine
    one-session multi-turn even if ids were renumbered."""
    present = [cid for cid in _MULTITURN_PREFERRED if cid in _CLIP_BY_ID]
    if len(present) >= 2:
        return present
    # Fallback: the longest contiguous group from one run, preserving order.
    by_run: dict[str, list[str]] = {}
    for c in _CLIPS:
        by_run.setdefault(c.get("run", ""), []).append(c["id"])
    best = max(by_run.values(), key=len, default=[])
    return best if len(best) >= 2 else []


def test_real_usage_multiturn(monkeypatch):
    """Drive an ordered multi-clip sequence from ONE session through a single
    runtime: every turn produces a final, and the EchoLLM response echoes that
    turn's own final (You said: ... <final> ...). A real multi-turn simulation
    through the real brain -- not isolated replays."""
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")

    clip_ids = _multiturn_clip_ids()
    replay_voice_driver.require_recorded_prerequisite(
        len(clip_ids) >= 2,
        "manifest has no multi-clip single-session sequence to replay",
    )

    scfg = replay_voice_driver.sherpa_config_or_skip()
    results = replay_voice_driver.run_turns(scfg, clip_ids)
    assert len(results) == len(clip_ids), (
        f"expected {len(clip_ids)} turns, got {len(results)}: "
        f"{[r.asr_final for r in results]!r}"
    )

    for clip_id, turn in zip(clip_ids, results):
        print(
            f"[multiturn {clip_id}] asr_final={turn.asr_final!r} "
            f"response={turn.response!r} first_audio_latency={turn.first_audio_latency}"
        )
        assert turn.asr_final_stamped, (
            f"{clip_id!r}: ASR_FINAL not stamped -- this turn never ran in the "
            f"multi-turn sequence"
        )
        assert turn.asr_final.strip(), (
            f"{clip_id!r}: turn ran but produced an empty final in the "
            f"multi-turn run"
        )
        from core.contract import is_stop_command

        if is_stop_command(turn.asr_final):
            assert turn.response == "", (
                f"{clip_id!r}: canonical stop must cancel locally without an "
                f"EchoLLM/TTS response, got {turn.response!r}"
            )
        else:
            # EchoLLM is deterministic: ordinary turns carry this final text.
            ok, overlap = _matches(turn.response, turn.asr_final)
            assert ok, (
                f"{clip_id!r}: EchoLLM response did not echo this turn's final "
                f"(overlap {overlap:.2f}). final={turn.asr_final!r} "
                f"response={turn.response!r}"
            )


# ---- (3) a sustained owner talk-over cuts the assistant off --------------------

@pytest.mark.skipif(
    not _BARGE_CASES,
    reason="recorded_voice_manifest.json has no explicit barge cases",
)
@pytest.mark.parametrize(
    "case",
    _BARGE_CASES,
    ids=[case["id"] for case in _BARGE_CASES],
)
def test_barge_overlap_cuts_assistant(monkeypatch, case):
    """Push a base utterance, let the assistant start speaking, then push the
    owner's real talk-over WHILE it speaks; the assistant must be cut off
    (BARGE_IN -> BARGE_IN_STOP fired -> barge_in_latency recorded).

    This is the only path that needs the InjectingInputStream (FileReplay is
    single-turn-at-a-time and can't model concurrent talk-over). Sounddevice
    streams and queries are replaced before startup, so it remains hardware-free.
    It uses ADR-0052's fake-stream detector profile, not production OS word-cut.
    """
    monkeypatch.setenv("SPEAKER_NO_LOCAL_CONFIG", "0")
    scfg = replay_voice_driver.sherpa_config_or_skip()

    base_id = case["base_id"]
    barge_id = case["barge_id"]
    assert base_id in _CLIP_BY_ID, f"barge case references unknown base {base_id!r}"
    assert barge_id in _BARGE_BY_ID, f"barge case references unknown clip {barge_id!r}"
    base_meta = _CLIP_BY_ID[base_id]
    barge_meta = _BARGE_BY_ID[barge_id]
    assert base_meta.get("run") == barge_meta.get("run") == case.get("run")
    assert float(barge_meta["end_sec"]) - float(barge_meta["start_sec"]) >= 0.2

    # The hash-pinned overlap waveform must contain sustained finite signal, not
    # a mislabeled empty window or the fake capture floor.
    import numpy as np

    barge_samples, barge_sr = replay_voice_driver.load_clip(barge_id)
    assert barge_sr == 16000
    assert barge_samples.size >= int(0.2 * barge_sr)
    assert np.isfinite(barge_samples).all()
    assert float(np.sqrt(np.mean(np.square(barge_samples)))) > 0.001

    # run_barge self-skips only when sounddevice is not installed; repository
    # import/config defects fail rather than disappearing as optional skips.
    turn = replay_voice_driver.run_barge(scfg, base_id, barge_id)

    print(
        f"[barge base={base_id} barge={barge_id}] "
        f"token={turn.target_turn_token} "
        f"owner_to_stop={turn.owner_to_stop_latency} "
        f"detector_to_stop={turn.barge_in_latency}"
    )
    expected = base_meta.get("expected_text", "")
    base_ok, overlap = _matches(turn.asr_final, expected)
    assert base_ok, (
        f"base ASR did not match the explicit same-session request "
        f"(overlap={overlap:.2f}): expected={expected!r} got={turn.asr_final!r}"
    )
    # The driver binds first consumed owner samples, BARGE_IN, the stop call, and
    # BARGE_IN_STOP to one exact metrics token after a floor-only no-cut control.
    assert turn.barged is True
    assert turn.barge_in_latency is not None, (
        f"talk-over did not cut the assistant: no BARGE_IN_STOP recorded "
        f"(base={base_id}, barge={barge_id})"
    )
    assert turn.barge_in_latency >= 0.0
    assert turn.owner_to_stop_latency is not None
    assert (
        turn.owner_to_stop_latency
        <= replay_voice_driver.RECORDED_OWNER_TO_STOP_MAX_SEC
    )
