from __future__ import annotations

import json

import pytest

from tools.bench import report, runner


def test_run_fake_produces_aligned_populated_samples():
    cases = [("greeting", "hello there"), ("time", "what time is it")]
    samples = runner.run_fake(cases)

    assert [s.name for s in samples] == ["greeting", "time"]
    # The scripted engine + echo LLM answer instantly, but the metrics path is
    # real: every turn must carry a measured first-audio latency.
    for s in samples:
        assert s.responded
        assert s.record.first_audio_latency is not None
        assert s.record.first_audio_latency >= 0


def test_summarize_and_calibration_shape():
    samples = runner.run_fake([("a", "what time is it"), ("b", "what time is it")])
    stats = report.summarize(samples)
    assert stats["turns"] == 2
    assert stats["first_audio_latency"]["count"] == 2
    assert stats["first_audio_latency"]["median"] is not None

    cal = report.calibration(stats, "phone")
    assert cal["spec"] == "Android phone (12 GB)"
    metrics = {row["metric"] for row in cal["rows"]}
    assert metrics == {"first_audio_latency", "barge_in_latency"}
    fa_row = next(r for r in cal["rows"] if r["metric"] == "first_audio_latency")
    # Instant fake latency is well under the phone budget -> "good".
    assert fa_row["status"] == "good"
    assert fa_row["modelled"] is not None  # specsim model number present


def test_write_reports_emits_html_and_json(tmp_path):
    samples = runner.run_fake([("a", "what time is it")])
    index = report.write_reports(tmp_path / "perf" / "phone", "phone", samples)
    assert index.exists()
    html = index.read_text(encoding="utf-8")
    assert "Speaker real-model latency" in html

    summary = json.loads((index.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "phone"
    assert summary["stats"]["turns"] == 1
    assert summary["calibration"]["spec"] == "Android phone (12 GB)"
    assert len(summary["turns"]) == 1


def test_markdown_summary_renders_table():
    samples = runner.run_fake([("a", "what time is it")])
    md = report.markdown_summary("phone", samples)
    assert "Speaker perf" in md
    assert "first_audio_latency" in md
    assert "| metric |" in md


def test_discover_fixtures_reads_metadata(tmp_path):
    np = pytest.importorskip("numpy")

    (tmp_path / "metadata.json").write_text(
        json.dumps({"cases": [{"name": "clip_a", "expectation": "callback"}]}),
        encoding="utf-8",
    )
    np.save(tmp_path / "clip_a.npy", np.zeros(10, dtype="float32"))
    np.save(tmp_path / "clip_b.npy", np.zeros(10, dtype="float32"))

    fixtures = runner.discover_fixtures(str(tmp_path))
    assert [f.name for f in fixtures] == ["clip_a", "clip_b"]
    assert fixtures[0].expectation == "callback"
    assert fixtures[1].expectation == ""  # not in metadata -> empty
