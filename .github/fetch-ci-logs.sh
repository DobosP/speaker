#!/usr/bin/env bash
# Fetch the latest GitHub Actions run status + failing-step logs for a branch.
#
# Read-only. Requires GH_TOKEN in the environment with scopes:
#   Actions: Read-only, Contents: Read-only  (fine-grained, repo dobosp/speaker)
# Usage: bash .github/fetch-ci-logs.sh [branch]
set -euo pipefail

OWNER=dobosp
REPO=speaker
BRANCH="${1:-claude/nice-planck-ZDr90}"
: "${GH_TOKEN:?GH_TOKEN is not set in the environment}"

API=https://api.github.com
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")

run_id=$(curl -sS "${auth[@]}" \
  "$API/repos/$OWNER/$REPO/actions/runs?branch=$BRANCH&per_page=1" \
  | python3 -c "import sys,json;r=json.load(sys.stdin).get('workflow_runs',[]);print(r[0]['id'] if r else '')")
[ -n "$run_id" ] || { echo "No workflow runs found for branch '$BRANCH'."; exit 1; }

echo "== run $run_id =="
curl -sS "${auth[@]}" "$API/repos/$OWNER/$REPO/actions/runs/$run_id" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d['display_title'], '|', d['status'], '|', d.get('conclusion'), '|', d['html_url'])"

status=$(curl -sS "${auth[@]}" "$API/repos/$OWNER/$REPO/actions/runs/$run_id" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
if [ "$status" != "completed" ]; then
  echo "Run still '$status' — re-run this script once it completes for logs."
  exit 0
fi

tmp=$(mktemp -d)
curl -sSL "${auth[@]}" "$API/repos/$OWNER/$REPO/actions/runs/$run_id/logs" -o "$tmp/logs.zip"
unzip -o -q "$tmp/logs.zip" -d "$tmp/logs"

echo "== failing-step log tails =="
grep -rliE "FAILURE|error:|exception|exit code" "$tmp/logs" 2>/dev/null | while read -r f; do
  echo "---- ${f#$tmp/logs/} ----"
  tail -n 50 "$f"
  echo
done
