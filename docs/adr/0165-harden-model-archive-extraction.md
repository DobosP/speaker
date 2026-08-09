# ADR-0165: Harden model archive extraction

Date: 2026-08-09
Status: accepted

## Decision

Route every tar-member write used by punctuation, SenseVoice, Parakeet, Kokoro, and KWS
model setup through one `_extract_tar_members` implementation. Preserve each
caller's release selection and layout contract: the punctuation/SenseVoice/Parakeet helper
keeps the first suffix match and flattens it, Kokoro keeps archive order and
strips one package root, and KWS keeps only its existing chunk-16/token/lexicon
selection and flattens those members. Archive symlinks, hardlinks, directories,
and special nodes remain ineligible; Kokoro still rejects parent components in
every regular member, while KWS rejects them only for selected members.

Make extraction fail closed at the destination boundary. Reject a symlinked,
junction-backed, or non-directory destination root and any such nested output
component. On POSIX, walk and create directories relative to retained
`O_DIRECTORY|O_NOFOLLOW` descriptors. On the cross-platform fallback, lstat
every stable component and reject every directory reparse tag even on Python
3.11, where `os.path.isjunction` is absent. Copy each selected member into a new exclusive sibling,
fsync it, and atomically replace the leaf. This replaces a pre-existing leaf
symlink or hardlink instead of truncating its external target, and a failed copy
leaves the prior leaf unchanged. Same-owner concurrent replacement during the
fallback component checks remains outside this local setup tool's threat model.

Before writing any member, reject output components with Windows drive/stream
syntax, device names, control/forbidden characters, or trailing dots/spaces so
Win32 normalization cannot escape or alias the selected tree. Require at most 20,000 archive entries, at most 4 GiB
for one selected member, and at most 16 GiB across the selected set. Read exactly
the declared member size. Keep duplicate-output compatibility: the first match
wins for the single-member helper, while later Kokoro/KWS members atomically
replace earlier members in archive order. Do not change model URLs, checksums,
selected model families, configuration wiring, runtime defaults, or live
authority.

## Context / why

Three near-identical extraction loops had accumulated in `tools/setup_models.py`.
Their archive-name traversal defenses differed, so a future correction could
easily reach only one copy. More importantly, every loop opened the final output
path directly. The single-member helper followed a pre-existing leaf symlink;
all three truncated through hardlinks; Kokoro/KWS could follow destination links
that resolved inside the tree; a symlinked destination root was accepted; and a
copy failure could leave a truncated model. The existing `TarInfo.isfile()`
filter prevented archive-authored link entries from being followed, but did not
protect the local destination tree.

Merely consolidating the path-based writes would preserve those defects in a
more reusable form. Rejecting every absolute or parent-bearing archive name
would be simpler, but would unnecessarily change the safe first-match flattening
contract and KWS's deliberate ignoring of unselected members. The shared helper
therefore owns output safety and bounds while caller selectors retain their
existing release semantics.

## Consequences

- One implementation now owns archive opening, bounds, normalized path mapping,
  no-follow directory admission, exact-size copying, and atomic leaf publication.
- Stable symlinked model destinations are no longer supported for these archive
  setup paths. Users must supply a real destination directory; this avoids
  ambiguous writes outside the selected model root.
- A pre-existing leaf link is replaced with a regular single-link file, leaving
  its former target unchanged. Per-member failures are rollback-safe. Multi-file
  package setup retains its existing final required-file validation and does not
  claim one atomic transaction across all distinct leaves.
- The generous finite limits admit the pinned releases without increasing
  memory pressure because payloads remain streamed in 1-MiB chunks.
- This is setup-only hardening. It does not download or run a model, allocate a
  GPU, touch an audio device, alter an application entry point, or provide live
  validation.

## Verification

- The focused archive suite passes 51 tests covering both mapping modes, archive link/special
  filtering, parent and separator normalization, duplicate order, unreadable
  members, destination root/component symlinks, leaf symlink/hardlink replacement,
  injected-copy rollback, temporary cleanup, and all three bounds.
- The combined setup doctor/Kokoro/KWS gate passes 230 tests; the all-setup
  adjacent gate passes 312 with two existing dependency warnings; APM/DTD passes
  6 under the low-priority single-thread gate.
- Scoped Ruff lint, new-test formatting, and whitespace checks are green. No
  archive was downloaded and no model, GPU, CPU-heavy inference, network, audio
  device, or live path ran.
