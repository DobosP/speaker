# ADR-0227: Refuse unqualified Windows enrollment persistence

Date: 2026-10-06
Status: accepted

## Decision

Refuse Windows enrollment mutation before the enrollment transaction inspects
supplied paths, builds a model/recorder or writes private state. Return the
existing core persistence-refusal code 5 and preparation/promotion CLI code 2.
Allow promotion module import/help without `fcntl`; absence of POSIX locking
still refuses mutation. Keep existing POSIX transactions and schema-v2 markers.
Require a separately qualified native ACL, handle identity, namespace durability
and stable locking implementation before enabling Windows enrollment persistence.

This adds an explicit platform admission boundary to ADR-0056/0066 and supersedes
neither. Their ownership/privacy/live-acceptance and commit guarantees survive.
The detailed native interface/security/marker plan remains a proposal in
[the qualification document](../windows_enrollment_persistence.md); it is not an
accepted backend or permission to use retained enrollment.

## Context / why

Windows lacks `fchmod`, `getuid` and `fcntl`. The current POSIX directory traversal
assumes a POSIX anchor. Windows stat ownership/mode placeholders and creation
timestamps cannot substitute for native SID/DACL/change-time checks. Previously
core enrollment could reach capture
before failing at persistence, and promotion failed at import. Conditional
imports alone would allow private reads before the missing backend was detected.

A synthetic empty-object probe showed native protected owner-only ACL creation,
file/directory flush calls and exclusive locking work on this host. It did not
qualify adversarial handle/security races, cross-process ownership or atomic
namespace crash durability. A successful FlushFileBuffers directory call does not
by itself establish ADR-0066's strict post-rename barrier. No Windows permission,
locking or durability fallback is accepted. Native API sources and exact pending
tests are in the qualification document.

## Consequences

Unsupported Windows transactions now explain the missing qualified backend
without opening their supplied enrollment/config paths or starting capture. Core
startup config loading and read-only existing enrollment loading are outside
this mutation admission guarantee. Auxiliary harness work before the guarded
writer is likewise outside this bounded transaction change.

Windows enrollment remains unavailable. ACL-based persistence needs its own
reviewed security/marker/durability decision and native synthetic/race/failure
qualification. Owner hardware/live gates remain separate. Linux/macOS private
mode, lineage, lock and outcome behavior remain unchanged by design; actual POSIX
regression tests were not run on this Windows host. No existing ADR is edited.
