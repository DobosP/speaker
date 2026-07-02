# Landing work from the Windows box (personal SSH key + PR flow)

How to push and land `speaker` work from this Windows machine. The repo guard
(`.claude/hooks/guard.ps1`, a Claude Code PreToolUse hook) intentionally stops an
autonomous session from pushing to `main` or touching the **work** identity — so
landing always goes through a **pull request**, never a direct `main` push.

## Git identity (SSH)

- **Remote:** `git@github-personal:DobosP/speaker.git`. The `github-personal`
  host alias in your SSH config maps to the **personal** key `id_ed25519_personal`
  (not the work key). Verify the key works without exposing it:
  `ssh -T git@github-personal` → should greet `DobosP`.
- **Allowed with this key:** feature-branch pushes, `fetch`/`pull`, and deleting
  remote *feature* branches (`git push origin --delete <branch>`).
- **Blocked by the guard (by design — do not evade):** the work key, the SSH
  config, the global git identity, `git config --global/--system`, and (explicit
  deny rule added 2026-07-02) any `git push` whose text targets `main`/`master`.
  The guard matches on the literal command text, so even *mentioning* those
  paths in a shell command is denied. Land via a PR.
- **Known gap:** a bare `git push` while `main` is checked out contains no
  `main` in its text, so the text-matching rule can't see it. The guard is a
  backstop, not the rule — the fleet no-push policy (`docs/adr/0007`) is what
  you follow.

## Why a PR (and not a direct push)

SSH is git transport only — it **cannot create or merge a GitHub pull request**.
Creating/merging a PR uses the GitHub **HTTPS API**, which needs either the `gh`
CLI (authenticated) or `$GIT_HUB_TOKEN` (see [`CREDENTIALS.md`](../CREDENTIALS.md)).
Keep the token in the environment; never hard-code or echo it.

## The loop: branch → push → PR → merge → delete

1. **Work on a short-lived branch**, then run the suite with the **venv** Python
   (git-bash `python` is system 3.10 and lacks `onnxruntime`/`sherpa_onnx`, which
   mass-errors collection):
   ```
   .venv\Scripts\python.exe -m pytest tests -q
   ```
2. **Commit and push the feature branch** (allowed by the guard):
   ```
   git push origin <branch>
   ```
3. **Open + merge the PR** (needs `gh` or `$GIT_HUB_TOKEN`). Either:

   **With `gh`:**
   ```
   gh pr create --base main --head <branch> --fill
   gh pr merge   <branch> --merge --delete-branch
   ```

   **With `curl` + token** (token read from env, never printed):
   ```
   # create:
   curl -sS -X POST -H "Authorization: Bearer $GIT_HUB_TOKEN" \
     https://api.github.com/repos/DobosP/speaker/pulls \
     -d '{"title":"<title>","head":"<branch>","base":"main"}'
   # merge (use the number the create call returns):
   curl -sS -X PUT -H "Authorization: Bearer $GIT_HUB_TOKEN" \
     https://api.github.com/repos/DobosP/speaker/pulls/<num>/merge \
     -d '{"merge_method":"merge"}'
   ```
   `tools/gh_admin.py` is the in-repo wrapper for privileged GitHub ops (dry-run by
   default, redacts the token); extend it if you want a one-command PR-merge.
4. **Delete the merged branch** (remote via SSH, local with `-d`):
   ```
   git push origin --delete <branch>     # remote (SSH; or gh --delete-branch did it)
   git branch -d <branch>                 # local (refuses if not merged)
   ```
5. **Reconcile local `main`** after the PR merges (all work is already on origin,
   so a hard reset is safe and loses nothing):
   ```
   git fetch origin
   git checkout main
   git reset --hard origin/main
   ```

## Gotcha worth remembering

- `core/runlog.py` prunes `logs/runs/` to `SPEAKER_KEEP_RUNS` (default 20) on each
  run finalize. A test that writes into the real `logs/runs/` can cull committed
  bundles. Set `SPEAKER_KEEP_RUNS=9999` before a full run, or `git restore logs/runs/`
  afterward (candidate fix: isolate the run-log dir in tests).
