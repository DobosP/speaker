from __future__ import annotations

import ctypes
import errno
import html
import json
import os
import secrets
import stat
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.wer import normalize, word_error_rate
from tools.specsim.simulate import (
    BARGE_IN_BUDGET,
    FIRST_AUDIO_BUDGET,
    SCENARIOS,
    classify,
    simulate_turn,
)
from tools.specsim.specs import CATALOG

from .runner import ProductionRun, TurnSample

_METRICS = (
    ("first_audio_latency", "First audio (speech end -> assistant speaks)"),
    ("endpoint_latency", "Endpoint (speech end -> ASR final)"),
    ("final_to_first_token", "ASR final -> first LLM token"),
    ("first_token_to_audio", "First token -> first audio"),
    ("barge_in_latency", "Barge-in (interrupt -> stop)"),
)

# Which specsim spec to calibrate a given bench profile against.
_PROFILE_TO_SPEC = {
    "phone": "Android phone (12 GB)",
    "desktop": "RTX 4090 Laptop",
}

_STATUS_COLORS = {"good": "#1e7e34", "ok": "#b8860b", "fail": "#c0392b", "n/a": "#777"}


def _values(samples: list[TurnSample], attr: str) -> list[float]:
    out = []
    for s in samples:
        v = getattr(s.record, attr)
        if v is not None:
            out.append(float(v))
    return out


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(samples: list[TurnSample]) -> dict[str, object]:
    """Per-metric count/median/p90/min/max plus a simple correctness tally."""
    stats: dict[str, object] = {}
    for attr, _label in _METRICS:
        vals = _values(samples, attr)
        stats[attr] = {
            "count": len(vals),
            "median": round(statistics.median(vals), 4) if vals else None,
            "p90": round(_percentile(vals, 0.9), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
        }
    responded = sum(1 for s in samples if s.responded)
    stats["turns"] = len(samples)
    stats["responded"] = responded
    return stats


def _modelled(spec_name: str) -> dict[str, Optional[float]]:
    spec = next((s for s in CATALOG if s.name == spec_name), None)
    quick = next((sc for sc in SCENARIOS if sc.name == "quick"), SCENARIOS[0])
    barge = next((sc for sc in SCENARIOS if sc.barge_in), None)
    if spec is None:
        return {"first_audio_latency": None, "barge_in_latency": None}
    fa = simulate_turn(spec, quick).first_audio_latency
    bi = simulate_turn(spec, barge).barge_in_stop if barge else None
    return {"first_audio_latency": fa, "barge_in_latency": bi}


def calibration(stats: dict[str, object], profile: str) -> dict[str, object]:
    """Measured median vs the specsim model, classified against the budgets."""
    spec_name = _PROFILE_TO_SPEC.get(profile, "Android phone (12 GB)")
    modelled = _modelled(spec_name)
    rows = []
    for attr, budget in (
        ("first_audio_latency", FIRST_AUDIO_BUDGET),
        ("barge_in_latency", BARGE_IN_BUDGET),
    ):
        measured = stats[attr]["median"] if isinstance(stats.get(attr), dict) else None  # type: ignore[index]
        rows.append(
            {
                "metric": attr,
                "measured_median": measured,
                "modelled": modelled.get(attr),
                "budget": list(budget),
                "status": classify(measured, budget) if measured is not None else "n/a",
            }
        )
    return {"spec": spec_name, "rows": rows}


# --- HTML rendering (self-contained, in the spirit of tools/specsim/report) ---
def _esc(text: object) -> str:
    return html.escape(str(text))


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}s" if value is not None else "—"


def render_html(profile: str, samples: list[TurnSample], stats: dict[str, object]) -> str:
    cal = calibration(stats, profile)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cal_rows = "".join(
        f"<tr><td>{_esc(r['metric'])}</td>"
        f"<td>{_fmt(r['measured_median'])}</td>"
        f"<td>{_fmt(r['modelled'])}</td>"
        f"<td>&le;{r['budget'][0]} / &le;{r['budget'][1]}</td>"
        f"<td><span class='chip' style='background:{_STATUS_COLORS.get(str(r['status']), '#777')}'>"
        f"{_esc(r['status'])}</span></td></tr>"
        for r in cal["rows"]
    )

    stat_rows = "".join(
        f"<tr><td>{_esc(label)}</td>"
        f"<td>{(stats[attr] or {}).get('count', 0)}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('median'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('p90'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('min'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('max'))}</td></tr>"
        for attr, label in _METRICS
    )

    sample_rows = "".join(
        f"<tr><td>{_esc(s.name)}</td><td>{_esc(s.expectation)}</td>"
        f"<td>{_esc(s.transcript)}</td>"
        f"<td>{'yes' if s.responded else 'no'}</td>"
        f"<td>{_fmt(s.record.first_audio_latency)}</td>"
        f"<td>{_fmt(s.record.barge_in_latency)}</td></tr>"
        for s in samples
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Speaker perf — {_esc(profile)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
 h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem}}
 table{{border-collapse:collapse;margin:.6rem 0;min-width:520px}}
 th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}}
 th{{background:#f4f4f4}}
 .chip{{color:#fff;border-radius:4px;padding:.1rem .5rem;font-size:.8rem}}
 .muted{{color:#666}}
</style></head><body>
<h1>Speaker real-model latency — profile <code>{_esc(profile)}</code></h1>
<p class="muted">{len(samples)} turn(s), {stats.get('responded', 0)} answered.
 Generated {generated}. Measured on this run's CPU — compare trends, and
 calibrate against the on-device model below rather than reading as phone-absolute.</p>

<h2>Calibration vs specsim budget ({_esc(cal['spec'])})</h2>
<table><tr><th>metric</th><th>measured median</th><th>modelled</th>
 <th>budget good/ok</th><th>status</th></tr>{cal_rows}</table>

<h2>Measured latency distribution</h2>
<table><tr><th>stage</th><th>n</th><th>median</th><th>p90</th><th>min</th><th>max</th></tr>
 {stat_rows}</table>

<h2>Per-turn</h2>
<table><tr><th>fixture</th><th>expectation</th><th>transcript</th>
 <th>answered</th><th>first audio</th><th>barge-in</th></tr>{sample_rows}</table>
</body></html>"""


def write_reports(out_dir: Path, profile: str, samples: list[TurnSample]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize(samples)
    cal = calibration(stats, profile)
    (out_dir / "index.html").write_text(render_html(profile, samples, stats), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "stats": stats,
                "calibration": cal,
                "turns": [s.as_dict() for s in samples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir / "index.html"


def markdown_summary(profile: str, samples: list[TurnSample]) -> str:
    """Compact summary for a CI job step ($GITHUB_STEP_SUMMARY)."""
    stats = summarize(samples)
    cal = calibration(stats, profile)
    lines = [
        f"## Speaker perf — `{profile}`",
        "",
        f"{stats['turns']} turn(s), {stats['responded']} answered.",
        "",
        "| metric | measured median | modelled | budget | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in cal["rows"]:
        lines.append(
            f"| {r['metric']} | {_fmt(r['measured_median'])} | {_fmt(r['modelled'])} "
            f"| <={r['budget'][0]} / <={r['budget'][1]} | {r['status']} |"
        )
    return "\n".join(lines) + "\n"


def _distribution(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "p50_seconds": (
            round(statistics.median(values), 4) if values else None
        ),
        "p95_seconds": round(_percentile(values, 0.95), 4) if values else None,
        "min_seconds": round(min(values), 4) if values else None,
        "max_seconds": round(max(values), 4) if values else None,
    }


def _contains_tokens(text: str, phrase: str) -> bool:
    tokens = tuple(normalize(text))
    wanted = tuple(normalize(phrase))
    return bool(wanted) and any(
        tokens[index : index + len(wanted)] == wanted
        for index in range(len(tokens) - len(wanted) + 1)
    )


def production_accuracy(samples: tuple[TurnSample, ...]) -> dict[str, object]:
    """Aggregate strict-corpus accuracy without retaining case text or IDs."""
    substitutions = insertions = deletions = reference_words = hypothesis_words = 0
    transcript_cases = exact_matches = nonempty_finals = responded = 0
    silence_cases = silence_false_finals = 0
    speech_negative_cases = speech_negative_nonempty_finals = 0
    target_mentions = target_hits = 0
    forbidden_mentions = forbidden_hits = 0
    for sample in samples:
        hypothesis_tokens = normalize(sample.transcript)
        nonempty_finals += bool(hypothesis_tokens)
        responded += bool(sample.responded)
        if sample.assertion == "transcript":
            transcript_cases += 1
            measured = word_error_rate(sample.reference, sample.transcript)
            substitutions += measured.substitutions
            insertions += measured.insertions
            deletions += measured.deletions
            reference_words += measured.ref_words
            hypothesis_words += measured.hyp_words
            exact_matches += normalize(sample.reference) == hypothesis_tokens
        elif sample.assertion == "silence":
            silence_cases += 1
            silence_false_finals += bool(hypothesis_tokens)
        elif sample.assertion == "speech_negative":
            speech_negative_cases += 1
            speech_negative_nonempty_finals += bool(hypothesis_tokens)
        for phrase in sample.commands:
            target_mentions += 1
            target_hits += _contains_tokens(sample.transcript, phrase)
        for phrase in sample.forbidden_commands:
            forbidden_mentions += 1
            forbidden_hits += _contains_tokens(sample.transcript, phrase)
    word_errors = substitutions + insertions + deletions
    return {
        "case_count": len(samples),
        "nonempty_final_count": int(nonempty_finals),
        "responded_count": int(responded),
        "transcript": {
            "case_count": transcript_cases,
            "exact_match_count": int(exact_matches),
            "word_error_rate": (
                round(word_errors / reference_words, 4)
                if reference_words
                else None
            ),
            "word_errors": word_errors,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "reference_words": reference_words,
            "hypothesis_words": hypothesis_words,
        },
        "silence": {
            "case_count": silence_cases,
            "nonempty_final_count": int(silence_false_finals),
        },
        "speech_negative": {
            "case_count": speech_negative_cases,
            "nonempty_final_count": int(speech_negative_nonempty_finals),
        },
        "commands": {
            "target_count": target_mentions,
            "target_hit_count": target_hits,
            "forbidden_target_count": forbidden_mentions,
            "forbidden_target_hit_count": forbidden_hits,
        },
    }


def production_latency(samples: tuple[TurnSample, ...]) -> dict[str, object]:
    """Only stages the accelerated replay transport actually observes."""
    final_to_token = [
        float(sample.record.final_to_first_token)
        for sample in samples
        if sample.record.final_to_first_token is not None
    ]
    token_to_clip = [
        float(sample.record.first_token_to_audio)
        for sample in samples
        if sample.record.first_token_to_audio is not None
    ]
    decode = [
        float(sample.replay_decode_seconds)
        for sample in samples
        if sample.replay_decode_seconds is not None
    ]
    idle = [
        float(sample.replay_to_idle_seconds)
        for sample in samples
        if sample.replay_to_idle_seconds is not None
    ]
    return {
        "complete-PCM→replay-decode-return": _distribution(decode),
        "complete-PCM→runtime-idle": _distribution(idle),
        "selected-final→token": _distribution(final_to_token),
        "token→TTS-clip-ready": _distribution(token_to_clip),
    }


def production_resources(system: dict, *, sample_interval_sec: float) -> dict[str, object]:
    """Reduce SystemMonitor output to honest, path-free sampled observations."""
    peak = system.get("peak", {}) if isinstance(system, dict) else {}
    if not isinstance(peak, dict):
        peak = {}
    samples = system.get("samples") if isinstance(system, dict) else None
    return {
        "sampling_interval_seconds": sample_interval_sec,
        "background_sample_count": samples if type(samples) is int else 0,
        "max_sampled_process_rss_mb": peak.get("proc_rss_mb"),
        "max_sampled_process_cpu_percent": peak.get("proc_cpu_percent"),
        "max_sampled_system_gpu_memory_used_mb": peak.get("gpu_mem_used_mb"),
        "max_sampled_system_gpu_util_percent": peak.get("gpu_util_percent"),
        "scope": {
            "process": "controller-process-including-local-sherpa-asr-and-tts",
            "gpu": "whole-system-sample-including-ollama-and-unrelated-processes",
            "sampling": "periodic-and-boundary-samples-may-miss-short-peaks",
            "limits": "observed-only-no-hard-resource-limit",
        },
    }


_PRODUCTION_LIMITATIONS = [
    "complete corpus PCM is available before each independent replay case",
    "each case gets a fresh engine/runtime/session while bound Ollama clients stay shared",
    "configured best-effort warm attempts must finish before timing but may have failed",
    "FileReplay feeds PCM faster than real time and opens no capture or playback device",
    "selected-final-to-token is post-ASR compute latency, not endpoint latency",
    "TTS clip-ready is offline synthesis completion at a null sink, not audible first audio",
    "capture, VAD timing, AEC, room echo, barge-in, and device playback are not covered",
    "the bound safety overlay forces loopback local-only LLMs and disables external providers",
    "durable memory is disabled; production local planner and control flow remain available",
    "resource values are sampled observations, not isolated process peaks or hard limits",
]


def production_payload(
    run: ProductionRun,
    corpus: dict[str, object],
    system: dict,
    *,
    sample_interval_sec: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "production-profile-whole-turn-file-replay",
        "privacy": "aggregate-only-path-and-transcript-free",
        "method": {
            "case_isolation": "fresh-engine-router-runtime-session-per-case",
            "llm_lifetime": "shared-bound-main-fast-ollama-clients",
            "warmup": "runtime-warm-ready-required-before-each-case-timing",
            "llm_transport": "loopback-local-only-fixed-evaluation-header",
            "runtime_logging": "suppressed-while-private-case-text-is-in-flight",
            "publication": "linux-renameat2-noreplace-private-directory",
        },
        "binding": {
            "device_profile": run.device_profile,
            "final_stt_profile": run.final_stt_profile,
            "final_stt_profile_sha256": run.final_stt_profile_sha256,
            "final_stt_profile_schema_version": run.final_stt_profile_schema_version,
            "production_config_sha256": run.production_config_sha256,
            "execution_config_sha256": run.execution_config_sha256,
            "replay_safety_schema_version": run.replay_safety_schema_version,
            "replay_safety_sha256": run.replay_safety_sha256,
            "active_audio_artifacts": {
                "sha256": run.active_artifacts_sha256,
                "configured_roots": run.active_artifact_roots,
                "files": run.active_artifact_files,
                "bytes": run.active_artifact_bytes,
            },
            "source_revision": run.source_revision,
            "runtime": run.runtime_identities,
            "ollama": {
                "sha256": run.ollama_identity_sha256,
                "roles": run.ollama_role_contracts,
            },
            "corpus": corpus,
        },
        "accuracy": production_accuracy(run.samples),
        "latency": production_latency(run.samples),
        "resources": production_resources(
            system,
            sample_interval_sec=sample_interval_sec,
        ),
        "limitations": list(_PRODUCTION_LIMITATIONS),
    }


def render_production_html(payload: dict[str, object]) -> str:
    binding = payload["binding"]
    accuracy = payload["accuracy"]
    latency = payload["latency"]
    assert isinstance(binding, dict) and isinstance(accuracy, dict)
    assert isinstance(latency, dict)
    transcript = accuracy["transcript"]
    assert isinstance(transcript, dict)
    rows = "".join(
        f"<tr><td>{_esc(name.replace('_', ' '))}</td>"
        f"<td>{_esc(values.get('count'))}</td>"
        f"<td>{_fmt(values.get('p50_seconds'))}</td>"
        f"<td>{_fmt(values.get('p95_seconds'))}</td></tr>"
        for name, values in latency.items()
        if isinstance(values, dict)
    )
    limits = "".join(
        f"<li>{_esc(item)}</li>" for item in payload.get("limitations", [])
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Speaker production-profile replay</title></head><body>
<h1>Speaker production-profile whole-turn replay</h1>
<p>Device <code>{_esc(binding.get('device_profile'))}</code>; final STT
<code>{_esc(binding.get('final_stt_profile'))}</code>.</p>
<p>Aggregate-only: no input path, case ID, reference text, recognized transcript,
or assistant reply is published.</p>
<h2>Accuracy</h2><p>WER {_esc(transcript.get('word_error_rate'))}; exact
{_esc(transcript.get('exact_match_count'))}/{_esc(transcript.get('case_count'))}.</p>
<h2>Replay latency</h2><table><tr><th>stage</th><th>n</th><th>p50</th><th>p95</th></tr>{rows}</table>
<h2>Limits</h2><ul>{limits}</ul></body></html>"""


def write_production_reports(
    out_dir: Path,
    run: ProductionRun,
    corpus: dict[str, object],
    system: dict,
    *,
    sample_interval_sec: float,
) -> tuple[Path, dict[str, object]]:
    payload = production_payload(
        run,
        corpus,
        system,
        sample_interval_sec=sample_interval_sec,
    )
    try:
        summary = (
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        rendered = render_production_html(payload).encode("utf-8")
    except (AssertionError, TypeError, ValueError, UnicodeError):
        raise _ProductionReportPublicationError() from None
    out_dir = _publish_private_report_directory(
        out_dir,
        {
            "summary.json": summary,
            "index.html": rendered,
        },
    )
    index = out_dir / "index.html"
    return index, payload


class _ProductionReportPublicationError(OSError):
    """Detail-free failure while publishing an aggregate production report."""


class _ProductionReportDestinationExists(FileExistsError):
    """The requested no-overwrite destination was already occupied."""


_REPORT_FILES = frozenset({"summary.json", "index.html"})
_RENAME_NOREPLACE = 1


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise _ProductionReportPublicationError()
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _owned_by_current_user(metadata: os.stat_result) -> bool:
    return not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid()


def _open_report_parent(path: Path, *, create: bool) -> tuple[Path, int]:
    """Open ``path.parent`` component-by-component without following symlinks."""
    selected = Path(os.path.abspath(path.expanduser()))
    if not selected.name or selected.parent == selected:
        raise _ProductionReportPublicationError()
    flags = _directory_open_flags()
    descriptor = -1
    try:
        descriptor = os.open(selected.anchor, flags)
        for component in selected.parent.parts[1:]:
            created = False
            child = -1
            try:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                        created = True
                    except FileExistsError:
                        # A concurrent creator is acceptable only if the
                        # no-follow open proves it installed a real directory.
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise _ProductionReportPublicationError()
                if created:
                    os.fchmod(child, 0o700)
                    metadata = os.fstat(child)
                    if (
                        stat.S_IMODE(metadata.st_mode) != 0o700
                        or not _owned_by_current_user(metadata)
                    ):
                        raise _ProductionReportPublicationError()
                    os.fsync(child)
                    os.fsync(descriptor)
            except BaseException:
                if child >= 0:
                    try:
                        os.close(child)
                    except OSError:
                        pass
                raise
            os.close(descriptor)
            descriptor = child
        return selected, descriptor
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _require_absent(parent_descriptor: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise _ProductionReportDestinationExists(
        "production report destination already exists"
    )


def _create_staging_directory(parent_descriptor: int) -> tuple[str, int, tuple[int, int]]:
    flags = _directory_open_flags()
    for _attempt in range(16):
        name = f".speaker-production-report-{os.getpid()}-{secrets.token_hex(12)}.tmp"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
            entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or not _owned_by_current_user(metadata)
                or _inode_identity(metadata) != _inode_identity(entry)
            ):
                raise _ProductionReportPublicationError()
            return name, descriptor, _inode_identity(metadata)
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                if stat.S_ISDIR(entry.st_mode):
                    os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    raise _ProductionReportPublicationError()


def _write_private_file_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> tuple[int, int]:
    if name not in _REPORT_FILES or not isinstance(payload, bytes) or not payload:
        raise _ProductionReportPublicationError()
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        os.fchmod(descriptor, 0o600)
        identity = _inode_identity(os.fstat(descriptor))
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if not isinstance(count, int) or count <= 0:
                raise _ProductionReportPublicationError()
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        retained = bytearray()
        while len(retained) < len(payload):
            chunk = os.read(descriptor, min(1024 * 1024, len(payload) - len(retained)))
            if not chunk:
                raise _ProductionReportPublicationError()
            retained.extend(chunk)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not _owned_by_current_user(metadata)
            or _inode_identity(metadata) != identity
            or _inode_identity(entry) != identity
            or bytes(retained) != payload
        ):
            raise _ProductionReportPublicationError()
        return identity
    except BaseException:
        if descriptor >= 0:
            try:
                owned = _inode_identity(os.fstat(descriptor))
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _inode_identity(current) == owned:
                    os.unlink(name, dir_fd=directory_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Linux atomic directory publication; fail closed without RENAME_NOREPLACE."""
    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
    except (AttributeError, TypeError, ValueError):
        raise _ProductionReportPublicationError() from None
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise _ProductionReportDestinationExists(
            "production report destination already exists"
        ) from None
    raise OSError(error, os.strerror(error))


def _cleanup_owned_report_directory(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int],
    file_identities: dict[str, tuple[int, int]],
) -> bool:
    descriptor = -1
    try:
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode) or _inode_identity(entry) != identity:
            return False
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)
        if _inode_identity(os.fstat(descriptor)) != identity:
            return False
        entries = set(os.listdir(descriptor))
        if not entries.issubset(_REPORT_FILES):
            return False
        for filename in entries:
            child = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
            expected = file_identities.get(filename)
            if (
                not stat.S_ISREG(child.st_mode)
                or child.st_nlink != 1
                or stat.S_IMODE(child.st_mode) != 0o600
                or not _owned_by_current_user(child)
                or (expected is not None and _inode_identity(child) != expected)
            ):
                return False
        for filename in entries:
            os.unlink(filename, dir_fd=descriptor)
        try:
            os.fsync(descriptor)
        except OSError:
            # Rollback must still remove the now-empty owned leaf when the
            # filesystem is already reporting a persistent sync fault.
            pass
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _inode_identity(current) != identity:
            return False
        os.rmdir(name, dir_fd=parent_descriptor)
        try:
            os.fsync(parent_descriptor)
        except OSError:
            # The caller is already failing publication.  The visible leaf is
            # gone, which is the cleanup invariant this best-effort path owns.
            pass
        return True
    except OSError:
        return False
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _publish_private_report_directory(
    path: Path,
    files: dict[str, bytes],
) -> Path:
    """Atomically publish one complete private directory without replacement."""
    if set(files) != _REPORT_FILES:
        raise _ProductionReportPublicationError()
    parent_descriptor = staging_descriptor = -1
    selected: Path | None = None
    staging_name: str | None = None
    staging_identity: tuple[int, int] | None = None
    file_identities: dict[str, tuple[int, int]] = {}
    try:
        selected, parent_descriptor = _open_report_parent(path, create=True)
        _require_absent(parent_descriptor, selected.name)
        staging_name, staging_descriptor, staging_identity = _create_staging_directory(
            parent_descriptor
        )
        for filename in ("summary.json", "index.html"):
            file_identities[filename] = _write_private_file_at(
                staging_descriptor,
                filename,
                files[filename],
            )
        os.fsync(staging_descriptor)
        entry = os.stat(
            staging_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _inode_identity(entry) != staging_identity:
            raise _ProductionReportPublicationError()

        # Re-open the requested path immediately before commit. This catches a
        # renamed/replaced parent chain while retaining the original descriptor
        # that owns the private staging directory.
        verified_descriptor = -1
        try:
            verified_selected, verified_descriptor = _open_report_parent(
                selected,
                create=False,
            )
            if (
                verified_selected != selected
                or _inode_identity(os.fstat(verified_descriptor))
                != _inode_identity(os.fstat(parent_descriptor))
            ):
                raise _ProductionReportPublicationError()
            _require_absent(verified_descriptor, selected.name)
        finally:
            if verified_descriptor >= 0:
                os.close(verified_descriptor)

        _rename_directory_noreplace(
            parent_descriptor,
            staging_name,
            selected.name,
        )
        current = os.stat(
            selected.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _inode_identity(current) != staging_identity:
            raise _ProductionReportPublicationError()
        os.fsync(parent_descriptor)
        return selected
    except _ProductionReportDestinationExists:
        raise
    except Exception:
        raise _ProductionReportPublicationError() from None
    finally:
        failure_active = sys.exc_info()[0] is not None
        if (
            failure_active
            and staging_identity is not None
            and parent_descriptor >= 0
        ):
            cleanup_names = []
            if selected is not None:
                cleanup_names.append(selected.name)
            if staging_name is not None and staging_name not in cleanup_names:
                cleanup_names.append(staging_name)
            for cleanup_name in cleanup_names:
                if _cleanup_owned_report_directory(
                    parent_descriptor,
                    cleanup_name,
                    staging_identity,
                    file_identities,
                ):
                    break
        if staging_descriptor >= 0:
            try:
                os.close(staging_descriptor)
            except OSError:
                pass
        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def production_markdown_summary(payload: dict[str, object]) -> str:
    binding = payload["binding"]
    accuracy = payload["accuracy"]
    latency = payload["latency"]
    assert isinstance(binding, dict) and isinstance(accuracy, dict)
    assert isinstance(latency, dict)
    transcript = accuracy["transcript"]
    assert isinstance(transcript, dict)
    lines = [
        f"## Speaker production replay — `{binding['device_profile']}`",
        "",
        f"Final STT: `{binding['final_stt_profile']}`; aggregate-only FileReplay.",
        f"WER: `{transcript['word_error_rate']}`; exact: "
        f"`{transcript['exact_match_count']}/{transcript['case_count']}`.",
        "",
        "| replay stage | n | p50 | p95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in latency.items():
        if isinstance(values, dict):
            lines.append(
                f"| {name} | {values['count']} | "
                f"{_fmt(values['p50_seconds'])} | {_fmt(values['p95_seconds'])} |"
            )
    lines.extend(
        [
            "",
            "TTS clip-ready is null-sink synthesis completion, not audible latency.",
        ]
    )
    return "\n".join(lines) + "\n"
