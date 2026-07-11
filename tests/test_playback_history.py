"""Receipt-ledger and resume-state tests with no audio-device dependency."""

from __future__ import annotations

from core.engine import PlaybackOutcome, PlaybackReceipt
from core.playback_history import PlaybackCommit, PlaybackHistory
from core.resume import ResumeConfig, ResumeTracker


def _receipt(
    fragment_id: str,
    outcome: PlaybackOutcome,
    safe_text_prefix: str = "",
    *,
    played_samples: int | None = None,
    total_samples: int | None = None,
) -> PlaybackReceipt:
    return PlaybackReceipt(
        fragment_id=fragment_id,
        outcome=outcome,
        safe_text_prefix=safe_text_prefix,
        played_samples=played_samples,
        total_samples=total_samples,
        output_sample_rate=16_000 if total_samples is not None else None,
    )


def test_completed_stream_commits_in_registration_order_after_all_receipts():
    history = PlaybackHistory()
    first = history.register(
        task_id="story",
        epoch=4,
        input_generation=4,
        followup_generation=4,
        text="First sentence.",
        remember=True,
        is_followup=False,
        streaming=True,
    )
    second = history.register(
        task_id="story",
        epoch=4,
        input_generation=4,
        followup_generation=4,
        text="Second sentence.",
        remember=True,
        is_followup=False,
        streaming=True,
    )

    assert history.close_stream("story", 4) == ()
    # Terminal callbacks may race; their arrival order must not reorder history.
    assert history.resolve(
        _receipt(second, PlaybackOutcome.COMPLETED, "Second sentence.")
    ).commits == ()
    resolution = history.resolve(
        _receipt(first, PlaybackOutcome.COMPLETED, "First sentence.")
    )

    assert resolution is not None
    assert resolution.commits == (
        PlaybackCommit(
            role="assistant",
            text="First sentence. Second sentence.",
            is_followup=False,
            schedule_followup=True,
            epoch=4,
            input_generation=4,
            followup_generation=4,
        ),
    )
    assert history.pending is False
    assert history.pending_fragments == 0


def test_interrupted_play_and_unheard_queued_fragment_commit_no_text():
    history = PlaybackHistory()
    started = history.register(
        task_id="answer",
        epoch=8,
        input_generation=8,
        followup_generation=8,
        text="This began playing.",
        remember=True,
        is_followup=False,
        streaming=True,
    )
    queued = history.register(
        task_id="answer",
        epoch=8,
        input_generation=8,
        followup_generation=8,
        text="This stayed queued.",
        remember=True,
        is_followup=False,
        streaming=True,
    )
    assert history.mark_started(started) == "This began playing."
    assert history.close_stream("answer", 8, interrupted=True) == ()

    started_resolution = history.resolve(
        _receipt(
            started,
            PlaybackOutcome.INTERRUPTED,
            played_samples=320,
            total_samples=3_200,
        )
    )
    queued_resolution = history.resolve(
        _receipt(
            queued,
            PlaybackOutcome.DROPPED,
            played_samples=0,
            total_samples=0,
        )
    )

    assert started_resolution is not None and started_resolution.played is True
    assert started_resolution.commits == ()
    assert queued_resolution is not None and queued_resolution.played is False
    assert queued_resolution.commits == ()
    assert history.pending is False


def test_interrupted_safe_partial_prefix_is_the_only_committed_text():
    history = PlaybackHistory()
    fragment_id = history.register(
        task_id="partial",
        epoch=2,
        input_generation=2,
        followup_generation=2,
        text="Safe words followed by uncertain audio.",
        remember=True,
        is_followup=False,
        streaming=True,
    )
    assert history.close_stream("partial", 2, interrupted=True) == ()

    resolution = history.resolve(
        _receipt(
            fragment_id,
            PlaybackOutcome.INTERRUPTED,
            "Safe words",
            played_samples=800,
            total_samples=4_000,
        )
    )

    assert resolution is not None
    assert resolution.safe_text_prefix == "Safe words"
    assert resolution.commits == (
        PlaybackCommit(
            role="assistant",
            text="Safe words",
            is_followup=False,
            schedule_followup=False,
            epoch=2,
            input_generation=2,
            followup_generation=2,
        ),
    )


def test_unknown_and_duplicate_receipts_are_idempotently_ignored():
    history = PlaybackHistory()
    fragment_id = history.register(
        task_id="once",
        epoch=1,
        input_generation=1,
        followup_generation=1,
        text="Remember once.",
        remember=True,
        is_followup=False,
        streaming=False,
    )
    terminal = _receipt(
        fragment_id,
        PlaybackOutcome.COMPLETED,
        "Remember once.",
    )

    resolution = history.resolve(terminal)

    assert resolution is not None and len(resolution.commits) == 1
    assert history.resolve(terminal) is None
    assert history.resolve(
        _receipt("unknown", PlaybackOutcome.COMPLETED, "Never registered.")
    ) is None
    assert history.pending is False


def test_delayed_old_receipt_cannot_commit_after_a_newer_reply():
    history = PlaybackHistory()
    old = history.register(
        task_id="old",
        epoch=1,
        input_generation=1,
        followup_generation=1,
        text="Old heard prefix.",
        remember=True,
        is_followup=False,
        streaming=False,
    )
    assert history.interrupt_all() == ()
    new = history.register(
        task_id="new",
        epoch=2,
        input_generation=2,
        followup_generation=2,
        text="New complete reply.",
        remember=True,
        is_followup=False,
        streaming=False,
    )

    # The new receipt is ready but held behind the older unresolved fragment.
    new_resolution = history.resolve(
        _receipt(new, PlaybackOutcome.COMPLETED, "New complete reply.")
    )
    assert new_resolution is not None and new_resolution.commits == ()

    old_resolution = history.resolve(
        _receipt(
            old,
            PlaybackOutcome.INTERRUPTED,
            "Old heard prefix.",
            played_samples=100,
            total_samples=500,
        )
    )
    assert old_resolution is not None
    assert [commit.text for commit in old_resolution.commits] == [
        "Old heard prefix.",
        "New complete reply.",
    ]


def test_user_marker_stays_between_older_and_later_assistant_groups():
    history = PlaybackHistory()
    old = history.register(
        task_id="old",
        epoch=1,
        input_generation=1,
        followup_generation=1,
        text="Old answer.",
        remember=True,
        is_followup=False,
        streaming=False,
    )
    assert history.stage_user(
        "user interruption",
        epoch=2,
        input_generation=2,
        followup_generation=2,
    ) == ()
    new = history.register(
        task_id="new",
        epoch=2,
        input_generation=2,
        followup_generation=2,
        text="New answer.",
        remember=True,
        is_followup=False,
        streaming=False,
    )

    assert history.resolve(
        _receipt(new, PlaybackOutcome.COMPLETED, "New answer.")
    ).commits == ()
    commits = history.resolve(
        _receipt(old, PlaybackOutcome.COMPLETED, "Old answer.")
    ).commits

    assert [(commit.role, commit.text) for commit in commits] == [
        ("assistant", "Old answer."),
        ("user", "user interruption"),
        ("assistant", "New answer."),
    ]


def test_followup_metadata_schedules_only_a_fully_completed_open_stream():
    history = PlaybackHistory()
    fragment_id = history.register(
        task_id="followup",
        epoch=6,
        input_generation=6,
        followup_generation=6,
        text="Would you like another?",
        remember=True,
        is_followup=False,
        streaming=True,
    )
    history.note_stream_metadata("followup", 6, is_followup=True)
    assert history.close_stream("followup", 6) == ()

    resolution = history.resolve(
        _receipt(
            fragment_id,
            PlaybackOutcome.COMPLETED,
            "Would you like another?",
        )
    )

    assert resolution is not None
    assert resolution.commits == (
        PlaybackCommit(
            role="assistant",
            text="Would you like another?",
            is_followup=True,
            schedule_followup=True,
            epoch=6,
            input_generation=6,
            followup_generation=6,
        ),
    )


def _resume_tracker() -> ResumeTracker:
    return ResumeTracker(ResumeConfig(enabled=True))


def test_started_fragment_with_empty_safe_prefix_is_still_resumable():
    tracker = _resume_tracker()
    tracker.note_query("Tell me a long story")
    tracker.stage_playback("started", "Words whose exact cutoff is unknown.")
    tracker.note_playback_started("started")
    tracker.note_cut()
    tracker.note_playback_receipt("started", "", played=True)

    prompt = tracker.resume_prompt("continue")

    assert prompt is not None
    assert "Tell me a long story" in prompt
    assert "[partial sentence; exact words unavailable]" in prompt
    assert "Words whose exact cutoff is unknown" not in prompt


def test_cut_before_async_played_receipt_retroactively_arms_resume():
    tracker = _resume_tracker()
    tracker.note_query("Tell me what happened")
    tracker.stage_playback("late", "A fragment with a delayed receipt.")

    tracker.note_cut()
    tracker.note_playback_receipt(
        "late",
        "A fragment",
        played=True,
    )

    prompt = tracker.resume_prompt("continue")
    assert prompt is not None
    assert "A fragment" in prompt


def test_zero_play_fragment_does_not_make_a_cut_resumable():
    tracker = _resume_tracker()
    tracker.note_query("Explain the queued answer")
    tracker.stage_playback("queued", "This never reached the sink.")
    tracker.note_playback_receipt("queued", "", played=False)
    tracker.note_cut()

    assert tracker.resume_prompt("continue") is None


def test_late_receipt_after_new_query_is_ignored():
    tracker = _resume_tracker()
    tracker.note_query("Old question")
    tracker.stage_playback("old", "Old answer that stayed in flight.")

    tracker.note_query("New question")
    tracker.note_playback_receipt(
        "old",
        "Old answer that stayed in flight.",
        played=True,
    )
    tracker.note_cut()

    assert tracker.resume_prompt("continue") is None


def test_resume_safe_text_keeps_fragment_registration_order():
    tracker = _resume_tracker()
    tracker.note_query("Tell me the ordered story")
    tracker.stage_playback("first", "First sentence.")
    tracker.stage_playback("second", "Second sentence.")

    tracker.note_playback_receipt("second", "Second sentence.", played=True)
    tracker.note_playback_receipt("first", "First sentence.", played=True)
    tracker.note_cut()

    prompt = tracker.resume_prompt("continue")
    assert prompt is not None
    assert prompt.index("First sentence.") < prompt.index("Second sentence.")
