"""Writes the run artifacts: an attributed timeline (JSON), a latency report, an
honest over-the-air STT grade, and a human-readable Markdown summary for grading.

The timeline is the heart of "differentiate the speakers": every event carries
``speaker`` (user vs assistant), the exact ``audio`` file that produced/played
it, timestamps, and -- for assistant turns -- the latency breakdown. For a user
turn, ``asr_final`` is what the assistant actually heard (the transcript that the
user's audio generated), closing the loop from played audio -> recognition ->
answer.

Two pieces of the acoustic full-duplex story live here as PURE, stdlib-only,
unit-testable functions (no audio/model deps):

* :func:`stt_score` grades what the mic heard against the scripted line, so the
  over-the-air STT quality is a number in [0, 1], not a vibe a human eyeballs.
* :func:`summarize_capture` turns the driver's continuous-capture observations
  (recording duration vs wall-clock, the capture-silent latch, partials produced
  while the user audio played) into an explicit full_duplex PASS/FAIL verdict.
"""
from __future__ import annotations

import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# A heard turn counts as "correct" at or above this blended-similarity score.
# It's a judgment call; grade.json always carries the raw score so a human can
# recalibrate without touching code.
HEARD_OK_THRESHOLD = 0.6
# The continuous-capture verdict treats the recording-vs-wallclock comparison as
# a tolerance band (device clock skew / resampling drift), not exact equality.
CAPTURE_COVERAGE_MIN = 0.85
# Barge-in is "working" when at least this fraction of the turns the scenario
# INTENDED to interrupt actually halted the assistant. A judgment call (a long
# answer should always be cuttable, but the short-answer race -- the barge lands
# after the answer already drained -- can legitimately miss a stop); grade.json
# carries the raw counts so a human can recalibrate. Named like
# HEARD_OK_THRESHOLD / CAPTURE_COVERAGE_MIN.
BARGE_STOP_RATE_MIN = 0.8


# --- over-the-air STT grading (pure) -----------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- so 'Paris.' and
    'paris' compare equal and only the words matter."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two normalized strings (order-insensitive, so a
    re-ordered-but-correct transcript still scores well)."""
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(wa), Counter(wb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(wb)
    recall = overlap / len(wa)
    return 2 * precision * recall / (precision + recall)


def stt_score(scripted: str, heard: Optional[str]) -> float:
    """Blend a difflib character-sequence ratio with a word-overlap F1 between the
    scripted line (truth) and what the mic heard (``asr_final``). Returns [0, 1]:
    1.0 for an exact match (after normalization), ~0 for total garble, monotonic
    in between. ``None``/empty heard -> 0.0 (the mic produced nothing)."""
    s = _normalize(scripted)
    h = _normalize(heard or "")
    if not s and not h:
        return 1.0
    if not h or not s:
        return 0.0
    seq = SequenceMatcher(None, s, h).ratio()
    f1 = _word_f1(s, h)
    return round(0.5 * seq + 0.5 * f1, 4)


def _grade_rows(events: list[dict]) -> list[dict]:
    """One row per user turn: scripted vs heard, the STT score, the heard-ok
    boolean, and (paired in order) the answer's first-audio latency."""
    rows: list[dict] = []
    assistant_lat = [
        _ms((e.get("latency") or {}).get("first_audio_latency"))
        for e in events if e.get("speaker") == "assistant"
    ]
    ai = 0
    for e in events:
        if e.get("speaker") != "user":
            continue
        scripted = e.get("text") or ""
        heard = e.get("asr_final")
        score = stt_score(scripted, heard)
        first_audio = assistant_lat[ai] if ai < len(assistant_lat) else None
        ai += 1
        rows.append({
            "turn": e.get("idx"),
            "scripted": scripted,
            "heard": heard,
            "stt_score": score,
            "heard_ok": score >= HEARD_OK_THRESHOLD,
            "first_audio_ms": first_audio,
            "capture": e.get("capture"),
        })
    return rows


def _grade_aggregate(rows: list[dict]) -> dict:
    scores = [r["stt_score"] for r in rows]
    n = len(scores)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "n_correct": sum(1 for r in rows if r["heard_ok"]),
        "stt_score_median": round(statistics.median(scores), 4),
        "stt_score_min": round(min(scores), 4),
        "stt_score_mean": round(statistics.fmean(scores), 4),
    }


# --- response-quality grading (pure) -----------------------------------------
#
# STT grading (above) asks "did the mic hear the user right?". Response grading
# asks the complementary question "did the assistant ANSWER right?" -- the thing
# the user actually cares about. It is graded against each Turn's ``expect`` /
# ``forbid`` (see scenarios.Turn):
#   * ``expect`` items are CONCEPTS the answer should contain; each may offer
#     alternatives with a "|" (e.g. "seven|7") and is satisfied if ANY matches.
#   * a turn's response score is the fraction of expect items satisfied (1.0 when
#     expect is empty -- a turn with only a ``forbid`` honesty check, or none).
#   * ``forbid`` items are substrings the answer must NOT contain -- the honesty
#     probes (a note/reminder/web claim the assistant can't fulfil). Any hit flags
#     the turn regardless of score.
# Like HEARD_OK_THRESHOLD, the ok bar is a judgment call and grade.json always
# carries the raw matched/missing/forbidden_hit so a human can recalibrate.

RESPONSE_OK_THRESHOLD = 0.6


def _alt_matches(alt: str, answer_norm: str, answer_tokens: set[str]) -> bool:
    """Does a single (already-normalized) alternative match the answer? A pure
    digit alternative ("4", "56") must match a WHOLE token (so "4" does not hit
    inside "40"); any alphabetic / multi-word alternative matches as a substring
    (so "moon" hits "moons", "freeze" hits "freezes", "da vinci" hits a phrase)."""
    alt = alt.strip()
    if not alt:
        return False
    if alt.isdigit():
        return alt in answer_tokens
    return alt in answer_norm


def _concept_matched(item: str, answer_norm: str, answer_tokens: set[str]) -> bool:
    """An expect/forbid ITEM matches if ANY of its "|"-separated alternatives
    matches. Each alternative is normalized the same way as the answer so
    punctuation/case never blocks a match (e.g. "fifty-six" -> "fifty six")."""
    for raw in item.split("|"):
        if _alt_matches(_normalize(raw), answer_norm, answer_tokens):
            return True
    return False


def response_score(
    expected: tuple[str, ...] | list[str],
    forbidden: tuple[str, ...] | list[str],
    answer: Optional[str],
) -> dict:
    """Grade one assistant answer against a turn's expect/forbid.

    Returns ``score`` (fraction of expect items satisfied; 1.0 when none),
    ``matched`` / ``missing`` (the expect items, split), ``forbidden_hit`` (the
    forbid items that appeared), and ``ok`` (score >= RESPONSE_OK_THRESHOLD AND no
    forbidden hit). An empty/None answer scores 0.0 when anything was expected (the
    assistant said nothing gradeable), but a forbid-only turn with no answer is
    vacuously ok (it cannot have made a false claim)."""
    expected = list(expected or [])
    forbidden = list(forbidden or [])
    answer_norm = _normalize(answer or "")
    answer_tokens = set(answer_norm.split())
    matched = [it for it in expected if _concept_matched(it, answer_norm, answer_tokens)]
    missing = [it for it in expected if it not in matched]
    forbidden_hit = [it for it in forbidden if _concept_matched(it, answer_norm, answer_tokens)]
    if not expected:
        score = 1.0
    elif not answer_norm:
        score = 0.0
    else:
        score = round(len(matched) / len(expected), 4)
    ok = bool(score >= RESPONSE_OK_THRESHOLD and not forbidden_hit)
    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "forbidden_hit": forbidden_hit,
        "ok": ok,
    }


def _answers_by_user_index(events: list[dict]) -> dict[int, str]:
    """Map the k-th user turn (0-based, in spoken order) to the assistant answer
    text that followed it (joined). Walks the ordered events: every assistant
    event attaches to the most-recent user turn. A turn with no answer maps to ""."""
    answers: dict[int, list[str]] = {}
    ui = -1
    for e in events:
        if e.get("speaker") == "user":
            ui += 1
            answers.setdefault(ui, [])
        elif e.get("speaker") == "assistant" and ui >= 0:
            txt = (e.get("text") or "").strip()
            if txt:
                answers[ui].append(txt)
    return {k: " ".join(v) for k, v in answers.items()}


def response_rows(scenario, events: list[dict]) -> list[dict]:
    """One row per GRADED turn (a turn with expect or forbid), pairing the scripted
    question + its expect/forbid with the assistant answer that followed and the
    response score. Turns with neither expect nor forbid are skipped (they are
    graded on STT + latency only). Robust to a short run: a scenario turn with no
    matching user event (the run timed out early) is skipped."""
    turns = getattr(scenario, "turns", ()) or ()
    user_events = [e for e in events if e.get("speaker") == "user"]
    answers = _answers_by_user_index(events)
    rows: list[dict] = []
    for k, turn in enumerate(turns):
        expect = tuple(getattr(turn, "expect", ()) or ())
        forbid = tuple(getattr(turn, "forbid", ()) or ())
        if not expect and not forbid:
            continue
        # An "immediately" (add-on/merge) or "barge_in" turn does NOT pair 1:1 with
        # the answer that follows its user event -- the merge produces one answer
        # after a LATER turn, and a barge flushes the INTERRUPTED prior answer right
        # after the barge line. Grading those would mis-attribute the answer, so
        # skip them (the response grade is for clean wait_for_response/pause turns).
        if getattr(turn, "timing", "") in ("immediately", "barge_in"):
            continue
        if k >= len(user_events):
            break
        answer = answers.get(k, "")
        graded = response_score(expect, forbid, answer)
        rows.append({
            "turn": user_events[k].get("idx"),
            "question": getattr(turn, "text", ""),
            "expect": list(expect),
            "forbid": list(forbid),
            "answer": answer,
            **graded,
        })
    return rows


def response_aggregate(rows: list[dict]) -> dict:
    """Headline response-quality verdict over the per-turn rows."""
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [r["score"] for r in rows]
    n_forbidden = sum(1 for r in rows if r["forbidden_hit"])
    return {
        "n": n,
        "n_ok": sum(1 for r in rows if r["ok"]),
        "response_score_median": round(statistics.median(scores), 4),
        "response_score_min": round(min(scores), 4),
        "response_score_mean": round(statistics.fmean(scores), 4),
        "n_forbidden_hits": int(n_forbidden),
        "verdict": "ok" if (n_forbidden == 0 and statistics.median(scores) >= RESPONSE_OK_THRESHOLD) else "FAIL",
    }


# --- continuous-capture / full-duplex verdict (pure) -------------------------


def summarize_capture(
    *,
    rec_seconds: float,
    wall_seconds: float,
    partials_during_user_total: int,
    capture_silent_warned: bool,
) -> dict:
    """Grade the full-duplex / always-on-capture story from the driver's
    observations. PASS requires all of:

    * the recorder covered ~the whole wall-clock session (it writes EVERY capture
      block, even during the assistant's own playback, so its duration is the
      ground truth that capture never paused);
    * no >5 s "capture silent" gap was ever latched (the audio thread stayed
      alive on its heartbeat cadence);
    * at least one partial appeared WHILE a user line was playing (the recognizer
      transcribed live, in parallel with output -> genuinely full-duplex).

    Coverage is a tolerance band, not exact equality, to avoid false FAILs from
    device clock skew / resampling drift on an otherwise healthy run."""
    coverage = (rec_seconds / wall_seconds) if wall_seconds > 0 else 0.0
    covered = coverage >= CAPTURE_COVERAGE_MIN
    transcribed_during_user = partials_during_user_total > 0
    ok = bool(covered and not capture_silent_warned and transcribed_during_user)
    return {
        "full_duplex": "ok" if ok else "FAIL",
        "ok": ok,
        "recording_seconds": round(float(rec_seconds), 2),
        "wall_seconds": round(float(wall_seconds), 2),
        "coverage": round(float(coverage), 3),
        "covered_whole_session": covered,
        "capture_silent_warned": bool(capture_silent_warned),
        "partials_during_user_total": int(partials_during_user_total),
        "transcribed_during_user": transcribed_during_user,
    }


# --- barge-in / interrupt grade (pure, inject-mode only) ---------------------
#
# Graded in INJECT MODE ONLY. A real two-stream acoustic barge-in (assistant
# audio + interrupter audio playing at once) is physically impossible on the
# reference box (exclusive ALSA hw -- a second output stream fails with "Device
# unavailable"), so the LOGIC is exercised by feeding the interrupter audio into
# the capture path during the assistant's speech. Read every verdict here as
# "barge-in LOGIC works", not "acoustic barge-in is proven".
#
# Inputs already produced upstream (reused, not re-derived):
#   * each assistant event carries ``barge_intended`` (driver: the PRECEDING user
#     line's timing == "barge_in") and ``interrupted`` (engine.stopped_after);
#   * its ``latency`` dict already includes ``barge_in_latency`` = BARGE_IN_STOP
#     minus BARGE_IN, stamped by runtime.metrics only when stop_speaking actually
#     abort()ed a live output stream.
#
# A "self-interrupt" -- the self-interruption-storm regression signal -- is an
# assistant turn that got interrupted (stopped_after, or a BARGE_IN_STOP latency
# landed) WITHOUT the scenario intending a barge before it. The no-barge CONTROL
# scenario must show ZERO of these.


def grade_barge_turns(events: list[dict]) -> list[dict]:
    """One row per assistant turn for the barge-in grade.

    ``stopped`` is the assistant event's ``interrupted`` flag (engine.stopped_after
    -- the primary, always-present "the answer was cut" signal). ``stop_ms`` is the
    barge->silence latency from BARGE_IN/BARGE_IN_STOP (None on the short-answer
    race, where BARGE_IN may have fired with no live stream to abort, so no
    BARGE_IN_STOP was stamped). ``self_interrupt`` flags a barge that fired with no
    intended barge before it -- keyed on ``stopped`` OR a present ``stop_ms`` so an
    aborted-but-mis-paired turn is still caught."""
    rows: list[dict] = []
    for e in events:
        if e.get("speaker") != "assistant":
            continue
        intended = bool(e.get("barge_intended"))
        stopped = bool(e.get("interrupted"))
        stop_ms = _ms((e.get("latency") or {}).get("barge_in_latency"))
        a_barge_fired = stopped or (stop_ms is not None)
        rows.append({
            "turn": e.get("idx"),
            "barge_intended": intended,
            "stopped": stopped,
            "stop_ms": stop_ms,
            "self_interrupt": bool(a_barge_fired and not intended),
        })
    return rows


def summarize_barge(rows: list[dict]) -> dict:
    """Headline barge-in verdict over the per-turn rows.

    ``stops_when_barged_rate`` = stopped / intended-barges (None when the scenario
    intended no barge -- the CONTROL case). ``stop_latency_ms_median`` is the median
    barge->silence latency over the turns that both intended a barge AND stopped.
    ``self_interrupt_count`` sums self-interrupts over ALL turns. Verdict is "ok"
    when there are zero self-interrupts AND (no barge was intended, OR the stop rate
    clears BARGE_STOP_RATE_MIN); else "FAIL"."""
    n_turns = len(rows)
    intended = [r for r in rows if r["barge_intended"]]
    n_intended = len(intended)
    n_stopped = sum(1 for r in intended if r["stopped"])
    self_interrupts = sum(1 for r in rows if r["self_interrupt"])
    stop_lats = [
        r["stop_ms"] for r in intended if r["stopped"] and r["stop_ms"] is not None
    ]
    rate = (n_stopped / n_intended) if n_intended else None
    if n_intended == 0:
        # Control scenario: nothing to stop, so the verdict keys ONLY off the
        # self-interruption-storm signal (zero self-interrupts == clean).
        verdict = "ok" if self_interrupts == 0 else "FAIL"
    else:
        verdict = "ok" if (rate >= BARGE_STOP_RATE_MIN and self_interrupts == 0) else "FAIL"
    return {
        "n_barge_turns": n_turns,
        "n_intended_barges": n_intended,
        "n_stopped": n_stopped,
        "stops_when_barged_rate": round(rate, 3) if rate is not None else None,
        "stop_latency_ms_median": round(statistics.median(stop_lats), 1) if stop_lats else None,
        "self_interrupt_count": int(self_interrupts),
        "verdict": verdict,
        "inject_mode_only": True,
    }


# --- artifact writers --------------------------------------------------------


def write_timeline(events: list[dict], out_dir: Path) -> Path:
    path = Path(out_dir) / "timeline.json"
    path.write_text(json.dumps(events, indent=2))
    return path


def _latencies(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        if e.get("speaker") != "assistant":
            continue
        lat = e.get("latency") or {}
        rows.append({
            "turn": e.get("idx"),
            "first_audio_ms": _ms(lat.get("first_audio_latency")),
            "endpoint_ms": _ms(lat.get("endpoint_latency")),
            "final_to_token_ms": _ms(lat.get("final_to_first_token")),
            "token_to_audio_ms": _ms(lat.get("first_token_to_audio")),
            "text": (e.get("text") or "")[:60],
        })
    return rows


def _ms(v):
    return round(v * 1000.0, 1) if isinstance(v, (int, float)) else None


def _pctls(values: list) -> dict:
    """A small percentile summary (min/p50/p90/p99/max/mean/n) over a list of
    numbers, with linear interpolation between order statistics. Empty -> {}. Used
    for both the per-scenario latency distribution and the pooled suite report, so
    a handful of real turns become an honest p50/p90/p99 instead of just a median."""
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not vals:
        return {}

    def pct(p: float) -> float:
        if len(vals) == 1:
            return round(vals[0], 1)
        k = (len(vals) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        if f == c:
            return round(vals[f], 1)
        return round(vals[f] + (vals[c] - vals[f]) * (k - f), 1)

    return {
        "n": len(vals),
        "min": round(vals[0], 1),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": round(vals[-1], 1),
        "mean": round(statistics.fmean(vals), 1),
    }


def write_latency_report(events: list[dict], out_dir: Path) -> Path:
    rows = _latencies(events)
    floor = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
    agg = {}
    if floor:
        d = _pctls(floor)
        # Keep the original keys (median/min/max/n) so existing readers/tests are
        # unchanged, and ADD the distribution percentiles + mean.
        agg = {
            "first_audio_ms_median": d["p50"],
            "first_audio_ms_min": d["min"],
            "first_audio_ms_max": d["max"],
            "n": d["n"],
            "first_audio_ms_p50": d["p50"],
            "first_audio_ms_p90": d["p90"],
            "first_audio_ms_p99": d["p99"],
            "first_audio_ms_mean": d["mean"],
        }
    # Per-stage distribution -- shows WHERE the first-audio time goes (the endpoint
    # trailing-silence wait usually dominates; the LLM and TTS are the rest).
    stages = {}
    for key in ("endpoint_ms", "final_to_token_ms", "token_to_audio_ms"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        d = _pctls(vals)
        if d:
            stages[key] = d
    report = {"per_turn": rows, "aggregate_first_audio": agg, "stages": stages}
    path = Path(out_dir) / "latency.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def write_grade(
    events: list[dict], out_dir: Path, *,
    capture: Optional[dict] = None, scenario=None,
) -> dict:
    """Emit grade.json -- the honest over-the-air STT grade (per turn + aggregate),
    the session full-duplex verdict, the barge-in grade, and (when ``scenario`` is
    given, so the per-turn expect/forbid is available) the response-quality grade.
    Returns the dict written."""
    rows = _grade_rows(events)
    resp_rows = response_rows(scenario, events) if scenario is not None else []
    report = {
        "per_turn": rows,
        "aggregate": _grade_aggregate(rows),
        "full_duplex": capture or {},
        "barge_in": summarize_barge(grade_barge_turns(events)),
        "response": {"per_turn": resp_rows, "aggregate": response_aggregate(resp_rows)},
    }
    path = Path(out_dir) / "grade.json"
    path.write_text(json.dumps(report, indent=2))
    return report


def write_summary(
    scenario, events: list[dict], out_dir: Path, *, voice: dict,
    capture: Optional[dict] = None,
) -> Path:
    """A Markdown summary a human reads to grade the run."""
    lines: list[str] = []
    lines.append(f"# Live validation: {scenario.name}\n")
    lines.append(f"**Capability:** {scenario.capability}\n")
    lines.append(f"**Goal:** {scenario.goal}\n")
    lines.append(
        f"**Synthetic-user voice:** speaker_id={voice.get('speaker_id')} "
        f"speed={voice.get('speed')} volume={voice.get('volume')}\n"
    )

    lines.append("\n## Conversation (attributed)\n")
    for e in events:
        who = "USER " if e["speaker"] == "user" else "ASSISTANT"
        t = f"{e.get('t_start', 0):6.2f}s"
        lines.append(f"- `{t}` **{who}** - {e.get('text','')!r}")
        if e.get("audio"):
            lines.append(f"    - audio: `{e['audio']}`")
        if e["speaker"] == "user" and e.get("asr_final") is not None:
            lines.append(f"    - heard as (asr_final): `{e['asr_final']}`")
        if e["speaker"] == "assistant" and e.get("latency"):
            la = e["latency"]
            lines.append(
                f"    - latency: first_audio={_ms(la.get('first_audio_latency'))}ms "
                f"(endpoint={_ms(la.get('endpoint_latency'))} | "
                f"final->token={_ms(la.get('final_to_first_token'))} | "
                f"token->audio={_ms(la.get('first_token_to_audio'))})"
            )

    rows = _latencies(events)
    lines.append("\n## Latency (ms)\n")
    lines.append("| turn | first_audio | endpoint | final->token | token->audio |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['turn']} | {r['first_audio_ms']} | {r['endpoint_ms']} | "
            f"{r['final_to_token_ms']} | {r['token_to_audio_ms']} |"
        )
    floor = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
    d = _pctls(floor)
    if d:
        lines.append(
            f"\n**first_audio distribution (n={d['n']}):** "
            f"p50 {d['p50']} | p90 {d['p90']} | p99 {d['p99']} "
            f"| min {d['min']} | max {d['max']} | mean {d['mean']} ms"
        )
        stage_bits = []
        for key, label in (("endpoint_ms", "endpoint"),
                           ("final_to_token_ms", "LLM"),
                           ("token_to_audio_ms", "TTS")):
            vals = [r[key] for r in rows if r.get(key) is not None]
            sd = _pctls(vals)
            if sd:
                stage_bits.append(f"{label} p50 {sd['p50']}")
        if stage_bits:
            lines.append(f"\n**where the time goes (p50):** " + " | ".join(stage_bits) + " ms")

    # Honest over-the-air STT grade.
    grade_rows = _grade_rows(events)
    agg = _grade_aggregate(grade_rows)
    lines.append("\n## STT accuracy (over-the-air)\n")
    if agg.get("n"):
        lines.append(
            f"**Headline:** median STT score {agg['stt_score_median']} "
            f"(min {agg['stt_score_min']}); heard correctly "
            f"{agg['n_correct']}/{agg['n']} turns (>= {HEARD_OK_THRESHOLD}).\n"
        )
    lines.append("| turn | scripted | heard | score | ok |")
    lines.append("|---|---|---|---|---|")
    for r in grade_rows:
        heard = (r["heard"] or "").replace("|", "/")
        scripted = (r["scripted"] or "").replace("|", "/")
        ok = "yes" if r["heard_ok"] else "NO"
        lines.append(
            f"| {r['turn']} | {scripted[:40]!r} | {heard[:40]!r} | "
            f"{r['stt_score']} | {ok} |"
        )

    # Response-quality grade (did the assistant ANSWER right, not just hear right).
    resp_rows = response_rows(scenario, events)
    if resp_rows:
        ragg = response_aggregate(resp_rows)
        lines.append("\n## Response quality (did the answer address the question)\n")
        lines.append(
            f"**Headline:** verdict {ragg.get('verdict')} -- median response score "
            f"{ragg.get('response_score_median')} (min {ragg.get('response_score_min')}); "
            f"{ragg.get('n_ok')}/{ragg.get('n')} turns ok (>= {RESPONSE_OK_THRESHOLD}); "
            f"forbidden-claim hits: {ragg.get('n_forbidden_hits')}.\n"
        )
        lines.append("| turn | question | expect | score | ok | forbidden_hit | answer |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in resp_rows:
            q = (r["question"] or "").replace("|", "/")[:34]
            exp = ", ".join(r["expect"]).replace("|", "/")[:24]
            ans = (r["answer"] or "").replace("|", "/").replace("\n", " ")[:46]
            fh = ", ".join(r["forbidden_hit"]).replace("|", "/")[:20] or "-"
            ok = "yes" if r["ok"] else "NO"
            lines.append(
                f"| {r['turn']} | {q!r} | {exp!r} | {r['score']} | {ok} | {fh} | {ans!r} |"
            )

    # Full-duplex / continuous-capture verdict.
    lines.append("\n## Full-duplex / continuous capture\n")
    if capture:
        verdict = capture.get("full_duplex", "n/a")
        lines.append(f"**full_duplex: {verdict}**\n")
        lines.append(
            f"- recording covered {capture.get('recording_seconds')}s of "
            f"{capture.get('wall_seconds')}s wall-clock "
            f"(coverage {capture.get('coverage')}, "
            f"covered_whole_session={capture.get('covered_whole_session')})"
        )
        lines.append(
            f"- capture-silent gap (>5s) seen: {capture.get('capture_silent_warned')} "
            f"(absence == the always-on proof)"
        )
        lines.append(
            f"- partials produced WHILE the user spoke: "
            f"{capture.get('partials_during_user_total')} "
            f"(transcribed_during_user={capture.get('transcribed_during_user')})"
        )
        lines.append(
            "- the recorder writes every capture block (even during the assistant's "
            "own playback), so its duration proves capture ran in+out at once."
        )
    else:
        lines.append("_no capture verdict recorded for this run._")

    # Barge-in / interrupt verdict (inject-mode only).
    barge_rows = grade_barge_turns(events)
    barge = summarize_barge(barge_rows)
    lines.append("\n## Barge-in / interrupt\n")
    lines.append(
        "_Graded in INJECT MODE ONLY: the interrupter audio is fed into the capture "
        "path during the assistant's speech. A real two-stream acoustic barge-in is "
        "physically impossible on the reference box (exclusive ALSA hardware), so "
        "this proves the barge-in LOGIC, not the acoustic path._\n"
    )
    rate = barge["stops_when_barged_rate"]
    rate_str = "n/a (no barge intended -- control)" if rate is None else f"{rate}"
    lines.append(
        f"**Verdict: {barge['verdict']}** -- stops-when-barged "
        f"{barge['n_stopped']}/{barge['n_intended_barges']} (rate {rate_str}), "
        f"median stop latency {barge['stop_latency_ms_median']}ms, "
        f"self-interrupts {barge['self_interrupt_count']} "
        f"(barge fired with NO intended barge -- the self-interruption-storm signal; "
        f"a clean run is 0).\n"
    )
    lines.append("| turn | barge_intended | stopped | stop_ms | self_interrupt |")
    lines.append("|---|---|---|---|---|")
    for r in barge_rows:
        bi = "yes" if r["barge_intended"] else "no"
        st = "yes" if r["stopped"] else "NO"
        si = "YES" if r["self_interrupt"] else "no"
        lines.append(
            f"| {r['turn']} | {bi} | {st} | {r['stop_ms']} | {si} |"
        )

    lines.append("\n## How to grade (from the scenario design)\n")
    lines.append(f"**Expected:** {scenario.expected_behavior}\n")
    if scenario.pass_signals:
        lines.append("\n**Pass signals:**")
        for s in scenario.pass_signals:
            lines.append(f"- [pass] {s}")
    if scenario.failure_modes:
        lines.append("\n**Failure modes:**")
        for s in scenario.failure_modes:
            lines.append(f"- [fail] {s}")

    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- consolidated suite report (cross-scenario dashboard) --------------------
#
# One report over MANY scenarios run as a battery: a pooled latency distribution
# (every turn across every scenario -> a real p50/p90/p99 + per-stage), pooled STT
# and response-quality numbers, and a per-scenario headline table. This is the
# "test everything + see latency + see how it responds" view; per-scenario
# summary.md files stay the drill-down.


def build_suite_report(runs: list[dict]) -> dict:
    """Aggregate per-scenario runs into one suite report dict.

    ``runs`` is a list of {"scenario": Scenario, "events": [...],
    "capture": dict|None}. Pure: recomputes every grade from events so the suite
    view can never drift from the per-scenario artifacts."""
    per_scenario: list[dict] = []
    pooled_first_audio: list[float] = []
    pooled_stage: dict[str, list[float]] = {
        "endpoint_ms": [], "final_to_token_ms": [], "token_to_audio_ms": []
    }
    pooled_stt: list[float] = []
    pooled_resp: list[dict] = []
    for run in runs:
        scenario = run.get("scenario")
        events = run.get("events") or []
        capture = run.get("capture")
        lat_rows = _latencies(events)
        stt_rows = _grade_rows(events)
        resp_rows = response_rows(scenario, events)
        barge = summarize_barge(grade_barge_turns(events))
        fa = [r["first_audio_ms"] for r in lat_rows if r["first_audio_ms"] is not None]
        pooled_first_audio.extend(fa)
        for key in pooled_stage:
            pooled_stage[key].extend(r[key] for r in lat_rows if r.get(key) is not None)
        pooled_stt.extend(r["stt_score"] for r in stt_rows)
        pooled_resp.extend(resp_rows)
        stt_agg = _grade_aggregate(stt_rows)
        resp_agg = response_aggregate(resp_rows)
        fa_d = _pctls(fa)
        per_scenario.append({
            "name": getattr(scenario, "name", "?"),
            "capability": getattr(scenario, "capability", ""),
            "n_turns": len(stt_rows),
            "first_audio_p50": fa_d.get("p50"),
            "first_audio_p90": fa_d.get("p90"),
            "stt_median": stt_agg.get("stt_score_median"),
            "response_median": resp_agg.get("response_score_median"),
            "response_ok": f"{resp_agg.get('n_ok', 0)}/{resp_agg.get('n', 0)}",
            "forbidden_hits": resp_agg.get("n_forbidden_hits", 0),
            "full_duplex": (capture or {}).get("full_duplex", "n/a"),
            "barge_verdict": barge.get("verdict") if barge.get("n_barge_turns") else "n/a",
        })
    pooled_resp_agg = response_aggregate(pooled_resp)
    return {
        "n_scenarios": len(runs),
        "latency": {
            "first_audio_ms": _pctls(pooled_first_audio),
            "stages": {k: _pctls(v) for k, v in pooled_stage.items() if v},
        },
        "stt": {
            "n": len(pooled_stt),
            "score_median": round(statistics.median(pooled_stt), 4) if pooled_stt else None,
            "score_min": round(min(pooled_stt), 4) if pooled_stt else None,
            "n_correct": sum(1 for s in pooled_stt if s >= HEARD_OK_THRESHOLD),
        },
        "response": pooled_resp_agg,
        "per_scenario": per_scenario,
    }


def write_suite_report(runs: list[dict], root: Path) -> dict:
    """Write SUITE.json + SUITE.md under ``root`` and return the report dict."""
    report = build_suite_report(runs)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "SUITE.json").write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# Live validation suite\n")
    lines.append(f"Ran **{report['n_scenarios']}** scenarios. "
                 "Per-scenario drill-down in each `<scenario>/summary.md`.\n")

    fa = report["latency"]["first_audio_ms"]
    if fa:
        lines.append("\n## Latency (pooled across every turn)\n")
        lines.append(
            f"**first_audio (n={fa['n']}):** p50 **{fa['p50']}** | p90 {fa['p90']} "
            f"| p99 {fa['p99']} | min {fa['min']} | max {fa['max']} | mean {fa['mean']} ms"
        )
        stages = report["latency"]["stages"]
        order = [("endpoint_ms", "endpoint (SPEECH_END->ASR_FINAL)"),
                 ("final_to_token_ms", "LLM (ASR_FINAL->first token)"),
                 ("token_to_audio_ms", "TTS (first token->first audio)")]
        lines.append("\n| stage | p50 | p90 | p99 | mean | n |")
        lines.append("|---|---|---|---|---|---|")
        for key, label in order:
            sd = stages.get(key)
            if sd:
                lines.append(f"| {label} | {sd['p50']} | {sd['p90']} | {sd['p99']} "
                             f"| {sd['mean']} | {sd['n']} |")

    stt = report["stt"]
    if stt.get("n"):
        lines.append("\n## Over-the-air STT (pooled)\n")
        lines.append(
            f"median {stt['score_median']} | min {stt['score_min']} | "
            f"heard-ok {stt['n_correct']}/{stt['n']} turns (>= {HEARD_OK_THRESHOLD})."
        )

    resp = report["response"]
    if resp.get("n"):
        lines.append("\n## Response quality (pooled)\n")
        lines.append(
            f"verdict **{resp.get('verdict')}** | median {resp.get('response_score_median')} "
            f"| {resp.get('n_ok')}/{resp.get('n')} turns ok (>= {RESPONSE_OK_THRESHOLD}) "
            f"| forbidden-claim hits {resp.get('n_forbidden_hits')}."
        )

    lines.append("\n## Per-scenario\n")
    lines.append("| scenario | turns | fa p50 | fa p90 | STT med | resp med | resp ok | forbid | full_duplex | barge |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in report["per_scenario"]:
        lines.append(
            f"| {s['name']} | {s['n_turns']} | {s['first_audio_p50']} | {s['first_audio_p90']} "
            f"| {s['stt_median']} | {s['response_median']} | {s['response_ok']} "
            f"| {s['forbidden_hits']} | {s['full_duplex']} | {s['barge_verdict']} |"
        )

    (Path(root) / "SUITE.md").write_text("\n".join(lines) + "\n")
    return report
