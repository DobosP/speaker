#!/usr/bin/env python3
"""
Generate regression tests from recorded live sessions.

Usage::

    python scripts/generate_session_tests.py [--session SESSION_ID] [--all] [--dry-run]

Each TTS turn recorded with ``python main.py --record`` becomes a pytest
test case.  The generated tests live in
``tests/recorded/test_<session_id>.py`` and are picked up automatically by
``pytest tests/test_recorded_sessions.py``.

Test classification
-------------------
``no_barge_in`` turns
    TTS played and no barge-in was triggered.  The replay test asserts that
    the pipeline does NOT fire barge-in when fed the same mic audio.  These
    catch regressions that re-introduce self-interruptions.

``barge_in_true_positive`` turns
    Barge-in fired AND heuristics suggest the user was genuinely speaking
    (e.g., voiced=True, low echo_sim, turn was followed by an STT result).
    The replay test asserts that barge-in DOES fire.

``barge_in_candidate`` turns
    Barge-in fired but the heuristics are uncertain (voiced=True but
    echo_sim > 0.20 with no subsequent STT confirmation).  These are saved
    for manual review — the generated test is marked ``xfail`` until
    annotated by a human.

Annotation
----------
Edit ``turn.json`` in the session directory and set::

    "annotation": "false_positive",  # or "true_positive" / "unknown"
    "annotation_reason": "assistant echo only"

Then re-run this script.  The ``annotation`` field overrides the automatic
classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = ROOT / "recordings"
GENERATED_TESTS_DIR = ROOT / "tests" / "recorded"

HEADER = '''"""
Auto-generated regression tests for session {session_id}.

DO NOT EDIT MANUALLY.  Re-generate with::

    python scripts/generate_session_tests.py --session {session_id}

Each test replays the mic audio received during a TTS turn through the barge-in
pipeline and asserts the expected behaviour (no self-interrupt / fires on user
speech).
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from tests.harness import AudioHarness, make_recorder

SESSION_DIR = Path(__file__).parent.parent.parent / "recordings" / "{session_id}"
GATE_SEC = 0.35     # must match production barge_in_min_delay_after_ref_sec
SR = 16_000
CHUNK_SEC = 1024 / SR
'''


def _classify_turn(turn: dict) -> str:
    """Return 'no_barge_in', 'true_positive', or 'candidate'."""
    annotation = turn.get("annotation", "")
    if annotation == "false_positive":
        return "no_barge_in"
    if annotation == "true_positive":
        return "true_positive"
    if annotation == "unknown":
        return "candidate"

    if not turn.get("barge_in_fired", False):
        return "no_barge_in"

    events = turn.get("barge_in_events", [])
    if not events:
        return "no_barge_in"

    # Heuristic: if any barge-in event had voiced=True and echo_sim < 0.15
    # it is likely a genuine user interrupt.
    for evt in events:
        if evt.get("voiced") and evt.get("echo_sim", 1.0) < 0.15:
            return "true_positive"

    return "candidate"


def _replay_audio_has_interrupt_evidence(turn_dir: Path, turn: dict) -> bool:
    """Return whether saved mic audio can plausibly replay the recorded event."""
    events = turn.get("barge_in_events", [])
    if not events:
        return False
    mic_path = turn_dir / "mic_16k.npy"
    if not mic_path.exists():
        return False
    try:
        mic = np.load(str(mic_path)).astype("float32")
    except Exception:
        return False
    if len(mic) == 0:
        return False
    max_chunk_rms = 0.0
    for offset in range(0, len(mic), 1024):
        chunk = mic[offset : offset + 1024]
        if len(chunk) < 1024:
            chunk = np.pad(chunk, (0, 1024 - len(chunk)))
        max_chunk_rms = max(
            max_chunk_rms,
            float(np.sqrt(np.mean(chunk**2))),
        )
    recorded_rms = max(float(evt.get("rms", 0.0)) for evt in events)
    return max_chunk_rms >= max(0.025, recorded_rms * 0.75)


def _write_no_barge_in_test(f, session_id: str, turn: dict):
    idx = turn["index"]
    text_snippet = turn.get("text", "")[:60].replace('"', "'").replace("\n", " ")
    echo_sim = turn.get("echo_sim_mean", 0.0)
    rms = turn.get("mic_rms_mean", 0.0)
    events = turn.get("barge_in_events", [])
    bi_note = ""
    if events:
        e = events[0]
        bi_note = (
            f" (false positive in session: RMS={e.get('rms', 0):.4f}, "
            f"echo_sim={e.get('echo_sim', 0):.2f}, voiced={e.get('voiced')})"
        )

    f.write(f'''
def test_turn_{idx:03d}_no_self_interrupt():
    """
    Turn {idx}: TTS played — no barge-in must fire during replay.

    Recorded TTS text: "{text_snippet}"
    Session mic RMS mean: {rms:.4f}, echo_sim mean: {echo_sim:.3f}{bi_note}

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_{idx:03d}" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_{idx:03d}" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn {idx}: self-interruption regression detected. "
        f"The pipeline fired barge-in {{len(interrupts)}} time(s) on TTS echo alone. "
        f"Last interrupt: {{interrupts[-1] if interrupts else 'n/a'}}"
    )
''')


def _write_true_positive_test(f, session_id: str, turn: dict):
    idx = turn["index"]
    text_snippet = turn.get("text", "")[:60].replace('"', "'").replace("\n", " ")
    events = turn.get("barge_in_events", [])
    first_evt = events[0] if events else {}

    f.write(f'''
def test_turn_{idx:03d}_barge_in_fires():
    """
    Turn {idx}: user spoke during TTS — barge-in must fire during replay.

    Recorded TTS text: "{text_snippet}"
    Barge-in: RMS={first_evt.get('rms', 0):.4f}, echo_sim={first_evt.get('echo_sim', 0):.2f},
              voiced={first_evt.get('voiced')}

    This turn was recorded with a confirmed user barge-in.  The test guards
    against regressions that would stop the user from interrupting the assistant.
    """
    tts_path = SESSION_DIR / "turn_{idx:03d}" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_{idx:03d}" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) > 0, (
        "Turn {idx}: barge-in regression — user speech no longer fires barge-in. "
        "The system would ignore a real user interrupt."
    )
''')


def _write_candidate_test(f, session_id: str, turn: dict):
    """Write an xfail test for uncertain barge-ins (needs human review)."""
    idx = turn["index"]
    text_snippet = turn.get("text", "")[:60].replace('"', "'").replace("\n", " ")
    events = turn.get("barge_in_events", [])
    first_evt = events[0] if events else {}

    f.write(f'''
@pytest.mark.xfail(
    reason=(
        "Turn {idx}: barge-in classification uncertain. "
        "Annotate turn.json with annotation=\\"false_positive\\" or \\"true_positive\\" "
        "then re-run: python scripts/generate_session_tests.py"
    ),
    strict=False,
)
def test_turn_{idx:03d}_barge_in_candidate():
    """
    Turn {idx}: barge-in fired but classification is uncertain (needs annotation).

    TTS text: "{text_snippet}"
    Barge-in: RMS={first_evt.get('rms', 0):.4f}, echo_sim={first_evt.get('echo_sim', 0):.2f},
              voiced={first_evt.get('voiced')}

    Edit recordings/{session_id}/turn_{idx:03d}/turn.json and set
    "annotation": "false_positive" or "true_positive", then regenerate tests.
    """
    tts_path = SESSION_DIR / "turn_{idx:03d}" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_{idx:03d}" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    # This test is xfail — we do not enforce an outcome until classified.
    # Remove xfail marker and pick the right assert once annotated.
    assert False, "Awaiting human annotation in turn.json"
''')


def generate_for_session(session_id: str, dry_run: bool = False) -> bool:
    session_dir = RECORDINGS_DIR / session_id
    meta_path = session_dir / "metadata.json"

    if not meta_path.exists():
        print(f"[SKIP] {session_id}: no metadata.json found", file=sys.stderr)
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    turns = meta.get("turns", [])
    if not turns:
        print(f"[SKIP] {session_id}: no turns recorded", file=sys.stderr)
        return False

    GENERATED_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    init_path = GENERATED_TESTS_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    out_path = GENERATED_TESTS_DIR / f"test_{session_id}.py"

    if dry_run:
        print(f"[DRY-RUN] Would write: {out_path}")
        for turn in turns:
            cls = _classify_turn(turn)
            print(f"  turn_{turn['index']:03d}: {cls}")
        return True

    counts = {"no_barge_in": 0, "true_positive": 0, "candidate": 0}
    with open(out_path, "w") as f:
        f.write(HEADER.format(session_id=session_id))
        for turn in turns:
            # Only generate tests for turns that have audio files
            turn_dir = session_dir / f"turn_{turn['index']:03d}"
            has_audio = (turn_dir / "tts_16k.npy").exists() and (turn_dir / "mic_16k.npy").exists()
            if not has_audio:
                continue

            cls = _classify_turn(turn)
            if cls == "true_positive" and not _replay_audio_has_interrupt_evidence(
                turn_dir,
                turn,
            ):
                cls = "candidate"
            counts[cls] += 1
            if cls == "no_barge_in":
                _write_no_barge_in_test(f, session_id, turn)
            elif cls == "true_positive":
                _write_true_positive_test(f, session_id, turn)
            else:
                _write_candidate_test(f, session_id, turn)

    total = sum(counts.values())
    print(
        f"[OK] {out_path.relative_to(ROOT)} — "
        f"{total} tests: {counts['no_barge_in']} no-barge-in, "
        f"{counts['true_positive']} true-positive, "
        f"{counts['candidate']} candidate (xfail)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--session",
        help="Session ID to generate tests for (e.g. session_20260208_143022)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate tests for ALL sessions in recordings/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    args = parser.parse_args()

    if not args.session and not args.all:
        # Default: process latest session
        sessions = sorted(RECORDINGS_DIR.glob("session_*/metadata.json"))
        if not sessions:
            print("No recorded sessions found in recordings/. Run main.py --record first.")
            sys.exit(1)
        latest = sessions[-1].parent.name
        print(f"No --session specified; using latest: {latest}")
        generate_for_session(latest, dry_run=args.dry_run)
        return

    if args.all:
        sessions = sorted(RECORDINGS_DIR.glob("session_*/metadata.json"))
        if not sessions:
            print("No recorded sessions found.")
            sys.exit(0)
        for meta_path in sessions:
            generate_for_session(meta_path.parent.name, dry_run=args.dry_run)
        return

    generate_for_session(args.session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
