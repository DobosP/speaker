# CREDENTIALS.md

The single source of truth for every credential this repo uses: where each one
comes from, what it unlocks, and how it is consumed. **This guide is the
reference; `CLAUDE.md` only points here.**

> **Golden rule (applies to every token below):** read the value from the
> environment at runtime. **Never** hard-code, echo, log, or commit a token, and
> never write it into a file — reference it only as `$VAR`. Tooling in this repo
> redacts tokens in all output (see `tools/gh_admin.py`); keep it that way.

## At a glance

| Credential | Where it comes from | Used by | Unlocks |
|---|---|---|---|
| `GIT_HUB_TOKEN` | session environment (this repo's web/CI sessions) | `tools/gh_admin.py`, ad-hoc `curl` | **Maximum repo access** — the privileged GitHub ops the session harness blocks. **Explicit authorization only** (see below) |
| `HUGGINGFACE_TOKEN` | session environment (Gemma license accepted) | `tools/bench/__main__.py` | Pulling gated Gemma weights at runtime (dev/bench) |
| `HF_TOKEN` | **GitHub Actions secret** | `perf.yml`, `publish-model.yml` | Same HuggingFace pull, but inside CI |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | session/host environment | `remote/token_server.py`, `core/app.py` | Minting LiveKit JWTs for the remote host + thin-client path |
| `DATABASE_URL` | developer `.env` (see `SETUP.md`) | `utils/memory.py` | Postgres-backed smart memory |

---

## `GIT_HUB_TOKEN` — maximum repo access (explicit authorization only)

> **Policy scope (fleet git standard 2026-06-24 — `AGENTS.md`, `docs/adr/0007`):**
> having this token is **not** authorization to use it. Agents never push, merge
> to `main`, delete branches, or run any of the admin operations below without
> Paul's **explicit ask**. The one sanctioned write path is an
> explicitly-authorized landing — feature branch → PR → merge — per
> [`docs/windows_landing_workflow.md`](docs/windows_landing_workflow.md), where
> `gh`/`$GIT_HUB_TOKEN` performs the PR create/merge the SSH transport cannot.

Ordinary git transport and GitHub reads/writes go through the **session
harness** (git proxy + repo-scoped GitHub MCP) or SSH with no stored
credential. The harness deliberately **blocks** a handful of admin operations.
`GIT_HUB_TOKEN` is the out-of-band key that performs exactly those, against
the GitHub REST API on `dobosp/speaker`:

| Operation | REST endpoint |
|---|---|
| Delete a branch | `DELETE /repos/dobosp/speaker/git/refs/heads/{branch}` |
| Trigger a workflow (`workflow_dispatch`) | `POST /repos/dobosp/speaker/actions/workflows/{file}/dispatches` |
| Re-run a whole run | `POST /repos/dobosp/speaker/actions/runs/{run_id}/rerun` |
| Re-run only failed jobs | `POST /repos/dobosp/speaker/actions/runs/{run_id}/rerun-failed-jobs` |
| List Actions secrets (names only) | `GET /repos/dobosp/speaker/actions/secrets` |
| Create / update an Actions secret | `PUT /repos/dobosp/speaker/actions/secrets/{name}` (encrypted — see below) |

### Use the helper (recommended)

`tools/gh_admin.py` wraps these in a dependency-free CLI. **It is dry-run by
default** — it prints the exact request (with the token redacted) and sends
nothing until you pass `--execute`; destructive ops also require `--yes`:

```bash
# See what would be sent (no network call, token redacted):
python -m tools.gh_admin runs --status failure
python -m tools.gh_admin dispatch perf.yml --ref main --input profile=phone

# Actually perform it:
python -m tools.gh_admin rerun-failed 1234567890 --execute
python -m tools.gh_admin delete-branch stale-feature --yes --execute
```

The helper reads `$GIT_HUB_TOKEN` from the environment and sends it only in the
`Authorization` header — it never prints it.

### Raw `curl` recipes

All calls take these headers:

```bash
AUTH=(-H "Authorization: Bearer $GIT_HUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28")

# Re-run the failed jobs of a run
curl -X POST "${AUTH[@]}" \
  https://api.github.com/repos/dobosp/speaker/actions/runs/RUN_ID/rerun-failed-jobs

# Trigger the perf benchmark on main
curl -X POST "${AUTH[@]}" \
  https://api.github.com/repos/dobosp/speaker/actions/workflows/perf.yml/dispatches \
  -d '{"ref":"main","inputs":{"profile":"phone"}}'

# Delete a branch
curl -X DELETE "${AUTH[@]}" \
  https://api.github.com/repos/dobosp/speaker/git/refs/heads/BRANCH_NAME
```

### Creating/updating an Actions secret (needs encryption)

Unlike the others, writing a secret requires sealing the value with the repo's
public key (libsodium / `PyNaCl`), so it is intentionally **left out of the
stdlib helper**. The flow:

```bash
# 1. Fetch the repo public key
curl "${AUTH[@]}" https://api.github.com/repos/dobosp/speaker/actions/secrets/public-key
# -> {"key_id": "...", "key": "<base64 public key>"}
```

```python
# 2. Seal the value (pip install pynacl)
import base64
from nacl import encoding, public
sealed = public.SealedBox(public.PublicKey(KEY, encoding.Base64Encoder)).encrypt(VALUE.encode())
encrypted_value = base64.b64encode(sealed).decode()
```

```bash
# 3. PUT it (use the key_id from step 1)
curl -X PUT "${AUTH[@]}" \
  https://api.github.com/repos/dobosp/speaker/actions/secrets/MY_SECRET \
  -d '{"encrypted_value":"...","key_id":"..."}'
```

> If a push returns `403 Permission denied`, the *session* was provisioned
> read-only. That is an environment permission, not a code problem, and
> `GIT_HUB_TOKEN` does not change it — surface it to the user.

---

## `HUGGINGFACE_TOKEN` vs `HF_TOKEN` — same purpose, two homes

Both are HuggingFace **read** tokens on an account that has accepted the Gemma
license at `huggingface.co/litert-community/Gemma3-1B-IT`. They are split only by
*where the code runs*:

- **`HUGGINGFACE_TOKEN`** — provided as an **env var in dev/web sessions**. Used
  to pull gated Gemma weights at runtime, e.g. the latency benchmark at
  `tools/bench/__main__.py` (which falls back to `HF_TOKEN` if the first is
  unset: `os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")`).
  Also usable directly: `hf_hub_download(..., token=os.environ["HUGGINGFACE_TOKEN"])`
  or `Authorization: Bearer $HUGGINGFACE_TOKEN` on `huggingface.co`.
- **`HF_TOKEN`** — the **GitHub Actions secret** form of the same token, injected
  into CI:
  - `perf.yml` → `env: HF_TOKEN: ${{ secrets.HF_TOKEN }}` (downloads Gemma GGUF +
    sherpa ONNX for the benchmark).
  - `publish-model.yml` → `env: HF: ${{ secrets.HF_TOKEN }}` (fetches the gated
    Gemma 3 1B and republishes it to the **public** `gemma-model` release).

> `android-apk.yml` needs **no** token: it downloads the model from that public
> `gemma-model` release, not from the gated HuggingFace repo. The shipped APK
> therefore carries no credential.

---

## `LIVEKIT_*` — remote host + thin-client path (optional)

Only needed for the `remote/` path (`pip install -r requirements-remote.txt`).
Read from the environment in `remote/token_server.py` (`LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET`, used to mint room JWTs) and `core/app.py` (`LIVEKIT_TOKEN`
for `--engine livekit`). `LIVEKIT_URL` points the browser/worker at the server.
Not required for any on-device path.

> Security note: `remote/token_server.py` mints a token for **any** identity/room
> as written — put real authentication in front of `/token` before exposing it.

---

## `DATABASE_URL` — Postgres memory store

Optional; the assistant runs in-memory without it. Full setup (PostgreSQL +
pgvector + `.env`) lives in [`SETUP.md`](SETUP.md).
