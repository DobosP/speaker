"""Tests for the GIT_HUB_TOKEN admin helper.

No network: every --execute path mocks urllib.request.urlopen, and dry-run paths
send nothing by design. The central guarantees are that the right request is
built per subcommand, the token is never printed, and destructive ops refuse
without --yes."""
from __future__ import annotations

import json
from unittest import mock

from tools import gh_admin

SECRET = "ghp_SUPER_SECRET_value_DO_NOT_PRINT"


def test_dispatch_dry_run_builds_request_without_sending(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    with mock.patch("urllib.request.urlopen") as urlopen:
        rc = gh_admin.main(
            ["dispatch", "perf.yml", "--ref", "main", "--input", "profile=phone"]
        )
    assert rc == 0
    urlopen.assert_not_called()
    out = capsys.readouterr().out
    assert "[dry-run] POST " in out
    assert "/repos/dobosp/speaker/actions/workflows/perf.yml/dispatches" in out
    assert '"profile": "phone"' in out
    assert '"ref": "main"' in out


def test_token_is_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    gh_admin.main(["runs", "--status", "failure"])
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "***redacted***" in out


def test_runs_builds_query(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    gh_admin.main(["runs", "--workflow", "tests.yml", "--status", "failure", "--limit", "5"])
    out = capsys.readouterr().out
    assert "GET " in out
    assert "/repos/dobosp/speaker/actions/workflows/tests.yml/runs?" in out
    assert "per_page=5" in out
    assert "status=failure" in out


def test_rerun_failed_path(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    gh_admin.main(["rerun-failed", "12345"])
    out = capsys.readouterr().out
    assert "/repos/dobosp/speaker/actions/runs/12345/rerun-failed-jobs" in out


def test_execute_sends_and_redacts(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    with mock.patch("urllib.request.urlopen") as urlopen:
        resp = urlopen.return_value.__enter__.return_value
        resp.read.return_value = b'{"ok": true}'
        resp.status = 204
        rc = gh_admin.main(["dispatch", "perf.yml", "--execute"])
    assert rc == 0
    urlopen.assert_called_once()
    req = urlopen.call_args[0][0]
    assert req.method == "POST"
    assert req.full_url.endswith("/actions/workflows/perf.yml/dispatches")
    assert json.loads(req.data) == {"ref": "main"}
    # The token rides only in the header object, never in printed output.
    assert "Bearer " + SECRET in req.headers.values()
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "-> HTTP 204" in out


def test_delete_branch_refuses_without_yes(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    with mock.patch("urllib.request.urlopen") as urlopen:
        rc = gh_admin.main(["delete-branch", "stale-feature", "--execute"])
    assert rc == 1
    urlopen.assert_not_called()
    assert "--yes" in capsys.readouterr().err


def test_delete_branch_dry_run_is_safe(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    with mock.patch("urllib.request.urlopen") as urlopen:
        rc = gh_admin.main(["delete-branch", "stale-feature"])
    assert rc == 0
    urlopen.assert_not_called()
    out = capsys.readouterr().out
    assert "DELETE " in out
    assert "/git/refs/heads/stale-feature" in out


def test_delete_branch_executes_with_yes(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    with mock.patch("urllib.request.urlopen") as urlopen:
        resp = urlopen.return_value.__enter__.return_value
        resp.read.return_value = b""
        resp.status = 204
        rc = gh_admin.main(["delete-branch", "stale-feature", "--yes", "--execute"])
    assert rc == 0
    req = urlopen.call_args[0][0]
    assert req.method == "DELETE"


def test_execute_without_token_errors(monkeypatch, capsys):
    monkeypatch.delenv("GIT_HUB_TOKEN", raising=False)
    with mock.patch("urllib.request.urlopen") as urlopen:
        rc = gh_admin.main(["rerun", "1", "--execute"])
    assert rc == 1
    urlopen.assert_not_called()
    assert "GIT_HUB_TOKEN" in capsys.readouterr().err


def test_dispatch_bad_input_errors(monkeypatch, capsys):
    monkeypatch.setenv("GIT_HUB_TOKEN", SECRET)
    rc = gh_admin.main(["dispatch", "perf.yml", "--input", "noequalsign"])
    assert rc == 1
    assert "KEY=VALUE" in capsys.readouterr().err
