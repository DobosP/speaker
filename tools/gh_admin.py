"""Privileged GitHub admin helper — the operations the session harness blocks.

This is the one place that actually *uses* ``$GIT_HUB_ACCESS_TOKEN`` (see
``CREDENTIALS.md``). Routine git and GitHub reads/writes go through the session
harness; the harness blocks a handful of admin operations (re-running and
dispatching Actions, deleting branches, reading Actions-secret metadata), and
this CLI performs exactly those against the GitHub REST API.

Dry-run by default: it prints the request it *would* send and sends nothing
until you pass ``--execute``. Destructive subcommands additionally require
``--yes``. The token is read from the environment and is never printed — the
Authorization header is redacted in all output.

    python -m tools.gh_admin runs --status failure
    python -m tools.gh_admin dispatch perf.yml --ref main --input profile=phone
    python -m tools.gh_admin rerun-failed 123456789 --execute
    python -m tools.gh_admin delete-branch stale-feature --yes --execute

Stdlib only (urllib), so it runs anywhere without extra deps. Writing an Actions
secret needs libsodium encryption and is intentionally left to CREDENTIALS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import IO, Any, Optional

REPO = "dobosp/speaker"
API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
TOKEN_ENV = "GIT_HUB_ACCESS_TOKEN"


class GitHubError(RuntimeError):
    """A request could not be built or the API returned an error."""


def _headers(token: Optional[str]) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _redacted(headers: dict[str, str]) -> dict[str, str]:
    safe = dict(headers)
    if "Authorization" in safe:
        safe["Authorization"] = "Bearer ***redacted***"
    return safe


def request(
    method: str,
    path: str,
    *,
    token: Optional[str],
    body: Optional[dict[str, Any]] = None,
    execute: bool = False,
    out: Optional[IO[str]] = None,
) -> Optional[Any]:
    """Build (and, when ``execute``, send) a GitHub REST request.

    Dry-run prints the request with the token redacted and returns ``None``.
    Execute sends it and returns the parsed JSON response (or ``{}``)."""
    out = out or sys.stdout
    url = f"{API_ROOT}{path}"
    headers = _headers(token)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    prefix = "" if execute else "[dry-run] "
    print(f"{prefix}{method} {url}", file=out)
    print(f"{prefix}headers: {_redacted(headers)}", file=out)
    if body is not None:
        print(f"{prefix}body: {json.dumps(body)}", file=out)

    if not execute:
        return None
    if not token:
        raise GitHubError(f"--execute needs ${TOKEN_ENV} set in the environment")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise GitHubError(f"{method} {path} -> HTTP {exc.code}: {detail[:300]}") from None

    print(f"-> HTTP {status}", file=out)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def cmd_runs(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    base = f"/repos/{REPO}/actions"
    path = f"{base}/workflows/{args.workflow}/runs" if args.workflow else f"{base}/runs"
    params = [f"per_page={args.limit}"]
    if args.status:
        params.append(f"status={args.status}")
    path += "?" + "&".join(params)
    return request("GET", path, token=token, execute=args.execute, out=out)


def cmd_rerun(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    return request(
        "POST", f"/repos/{REPO}/actions/runs/{args.run_id}/rerun",
        token=token, execute=args.execute, out=out,
    )


def cmd_rerun_failed(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    return request(
        "POST", f"/repos/{REPO}/actions/runs/{args.run_id}/rerun-failed-jobs",
        token=token, execute=args.execute, out=out,
    )


def cmd_dispatch(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    body: dict[str, Any] = {"ref": args.ref}
    inputs: dict[str, str] = {}
    for item in args.input or []:
        if "=" not in item:
            raise GitHubError(f"--input expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        inputs[key] = value
    if inputs:
        body["inputs"] = inputs
    return request(
        "POST", f"/repos/{REPO}/actions/workflows/{args.workflow}/dispatches",
        token=token, body=body, execute=args.execute, out=out,
    )


def cmd_delete_branch(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    if args.execute and not args.yes:
        raise GitHubError("delete-branch is destructive; pass --yes to confirm")
    return request(
        "DELETE", f"/repos/{REPO}/git/refs/heads/{args.branch}",
        token=token, execute=args.execute, out=out,
    )


def cmd_secrets_list(args: argparse.Namespace, token: Optional[str], out: IO[str]) -> Optional[Any]:
    return request(
        "GET", f"/repos/{REPO}/actions/secrets",
        token=token, execute=args.execute, out=out,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.gh_admin",
        description=(
            f"Privileged GitHub admin ops on {REPO} using ${TOKEN_ENV}. "
            "Dry-run by default — pass --execute to actually send."
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--execute", action="store_true",
        help="actually send the request (default: dry-run, prints only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", parents=[common], help="list workflow runs (read-only)")
    p_runs.add_argument("--workflow", help="scope to a workflow file/id, e.g. tests.yml")
    p_runs.add_argument("--status", help="filter by status/conclusion, e.g. failure")
    p_runs.add_argument("--limit", type=int, default=20, help="max runs (per_page)")
    p_runs.set_defaults(func=cmd_runs)

    p_rerun = sub.add_parser("rerun", parents=[common], help="re-run an entire workflow run")
    p_rerun.add_argument("run_id")
    p_rerun.set_defaults(func=cmd_rerun)

    p_rerunf = sub.add_parser(
        "rerun-failed", parents=[common], help="re-run only the failed jobs of a run"
    )
    p_rerunf.add_argument("run_id")
    p_rerunf.set_defaults(func=cmd_rerun_failed)

    p_disp = sub.add_parser("dispatch", parents=[common], help="trigger a workflow_dispatch")
    p_disp.add_argument("workflow", help="workflow file/id, e.g. perf.yml")
    p_disp.add_argument("--ref", default="main", help="git ref to run on (default: main)")
    p_disp.add_argument(
        "--input", action="append", metavar="KEY=VALUE", help="workflow input (repeatable)"
    )
    p_disp.set_defaults(func=cmd_dispatch)

    p_del = sub.add_parser(
        "delete-branch", parents=[common], help="delete a branch (destructive; needs --yes)"
    )
    p_del.add_argument("branch")
    p_del.add_argument("--yes", action="store_true", help="confirm the deletion")
    p_del.set_defaults(func=cmd_delete_branch)

    p_sec = sub.add_parser(
        "secrets-list", parents=[common], help="list Actions secret names (metadata only)"
    )
    p_sec.set_defaults(func=cmd_secrets_list)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(TOKEN_ENV)
    try:
        args.func(args, token, sys.stdout)
    except GitHubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
