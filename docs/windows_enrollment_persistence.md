# Windows enrollment persistence qualification

Last verified: 2026-10-06 (synthetic Windows API capability/admission checks only).

[ADR-0227](adr/0227-refuse-unqualified-windows-enrollment-persistence.md) owns the
current admission boundary. Windows mutation is refused before the transaction
inspects supplied paths or constructs a recorder. This document is a concrete
**unimplemented adapter proposal**, not a Windows enrollment/promotion pass.
[ADR-0056](adr/0056-isolate-v5-enrollment-candidates.md) and
[ADR-0066](adr/0066-promote-accepted-v5-enrollment-without-replacing-v4.md) retain the transaction contract.

## Source scope and observed gap

The affected seams are `core/enroll.py` (snapshot/atomic JSON/config writes),
`tools/prepare_enrollment.py` (backup/reservation/final isolated-config publication),
and `tools/promote_enrollment.py` (ownership, private lock, strict durability and
commit classification). Their POSIX implementation remains in place.

The Windows admission helper is `core/_enrollment_persistence.py`. Core enrollment
returns its existing persistence-refusal code 5. Preparation/promotion CLI refusal
returns 2; promotion imports and `--help` work without `fcntl`. Direct config and
embedding writes raise a caught `ValueError` subtype. Existing enrollment loading,
frontend/identity policy and owner acceptance are outside this mutation guard.
`core.app` can load the local config before dispatching enrollment; the guard
makes no broader claim about startup/config-loading IO. Auxiliary noise-stress
harnesses can perform earlier work before calling the guarded writer.

`chmod(0600)`, Windows `st_uid`/`st_gid` placeholders and path-based ACL tightening
cannot prove private ownership. `_safe_directory_chain` also assumes a POSIX
anchor. Windows junctions and other reparse tags require explicit handling.
No missing native operation is replaced by a best-effort success.

## Proposed native backend

Add a separately reviewed `core/_enrollment_windows.py` behind the admission seam
only after qualification. Keep the POSIX backend/markers byte-compatible. The
following interface is a source proposal; it is not installed or selectable:

```python
@dataclass(frozen=True)
class WindowsSecurity:
    owner_sid: bytes                 # copied SID, compare with TokenUser
    dacl: bytes                      # canonical validated ACE/control data
    protected: bool

@dataclass(frozen=True)
class WindowsFileState:
    volume: int
    file_id: bytes
    size: int
    write_time: int
    change_time: int
    links: int
    attributes: int
    security: WindowsSecurity
    sha256: bytes
    ancestors: tuple[WindowsDirectoryState, ...]

class WindowsEnrollmentBackend(Protocol):
    def admit_local_volume(self, directory: Path) -> None: ...
    def snapshot(self, path: Path) -> WindowsFileState: ...
    def revalidate(self, path: Path, expected: WindowsFileState) -> None: ...
    def create_private(self, parent: Path, name: str) -> PrivateHandle: ...
    def flush_file(self, handle: PrivateHandle) -> None: ...
    def publish_absent(self, handle: PrivateHandle, path: Path) -> None: ...
    def replace_bound(self, handle: PrivateHandle, path: Path,
                      expected: WindowsFileState) -> None: ...
    def sync_namespace(self, directory: Path) -> None: ...
    def config_lock(self, primary: Path) -> ContextManager[PrivateHandle]: ...
```

The types refer to proposed native objects, not Python stat substitutes.
`PrivateHandle` owns one non-inheritable handle and exact cleanup; directory
snapshots include native identity and security. Byte/digest/SID values stay in
private state and are never printed. Unknown capabilities raise admission errors.

1. **Private creation and ownership.** Obtain `TokenUser` from the effective
   process/thread security context. Create files with `CreateFileW(CREATE_NEW)`
   and directories with `CreateDirectoryW`, supplying a security descriptor at
   creation. The proposed v1 policy is an explicit current-user owner and a
   present, non-null, protected DACL containing one non-inherited full-access
   allow ACE for that SID. Reject other/unknown/object/callback ACEs and ACL
   query failures. Existing objects are inspected, never silently repaired.
   A parent-directory rule must reject non-owner rights to add/delete/rename
   children or change its security; private file ACLs alone do not protect names.
   Details: [CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew),
   [GetSecurityInfo](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo),
   [token classes](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ne-winnt-token_information_class).
2. **Path and handle binding.** Admit local volumes with persistent ACL support;
   reject UNC/device/alternate-stream names and unsupported volume capabilities.
   Traverse from the actual drive anchor. Open and retain ancestor handles
   without delete sharing through commit, reject every reparse point, and bind
   each component to its native volume/file ID and security. Read bytes only
   through a validated no-follow handle. Snapshot/revalidate identity, link
   count, native write/change times, ACL/control data, size, bytes and ancestors
   around every read and before publication. Retain ADR-0066's documented
   non-cooperating check/replace limitation; these steps need adversarial tests.
   Details: [file identity](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle),
   [native change time](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info),
   [volume capabilities](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getvolumeinformationw).
3. **Private marker lineage.** Propose schema 3 with an explicit backend tag and
   native snapshots for source, backup, config and empty reservation. Refuse
   schema-2 POSIX uid/gid/mode fields as Windows ownership evidence. Retain exact
   model, sample-rate, dimension, passes, frontend-v5, content/disjointness and
   reservation-replacement checks. Do not upgrade retained private state by
   reading or rewriting it automatically. This schema/policy needs its own ADR
   before implementation; ADR-0227 accepts only the current refusal.
4. **Stable lock.** Exclusively create/open the protected adjacent lock file;
   verify owner/ACL/one link and handle/path identity. Use
   `LockFileEx(EXCLUSIVE | FAIL_IMMEDIATELY)` over byte 0 and hold the same
   handle through config publication and required namespace durability. Keep
   the lock file stable and separate from replaced config objects. Contention
   refuses; closing/terminating the owner releases the lock. This coordinates
   promoters and cannot exclude memory-mapped/noncooperating writers.
   Details: [LockFileEx](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex).
5. **Durability and outcomes.** Flush writable temporary handles, verify copied
   bytes/security, and publish backup/accepted names without clobber. Retain the
   required strict post-publication namespace barrier; prepare publishes isolated
   config last, promote commits its pointer last. A same-volume handle rename or
   `MoveFileExW(WRITE_THROUGH)` is a candidate requiring qualification, never
   `COPY_ALLOWED`. `ReplaceFileW` preserves the old DACL and its write-through
   flag is unsupported, so it is not an assumed private publish primitive.
   Details: [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw),
   [ReplaceFileW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew).

`FlushFileBuffers` on a writable synthetic directory succeeded on this host.
Microsoft documents file-buffer flushing and an administrator-only volume flush;
the cited page does not establish the exact post-rename namespace guarantee.
MoveFileEx documents disk completion with an explicit copy/delete flush guarantee.
Treating these as ADR-0066's strict directory barrier is an **unqualified inference**.
Successful calls alone do not establish power-loss durability. A production
backend remains blocked on a reviewed, supported-filesystem durability contract;
no admin volume handle was opened. Sources:
[FlushFileBuffers](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers),
[file caching](https://learn.microsoft.com/en-us/windows/win32/fileio/file-caching).

Preserve promotion exit 2 before accepted/config commit, 3 only for verified
independent durable accepted bytes plus an unchanged inactive primary, and 4 for
ambiguous publication/cleanup/required durability. Never report 0 before both
accepted and pointer publication barriers complete. After success perform no
fallible read that could relabel a committed result as refusal.

## Exact unresolved tests before enabling the backend

All cases below are **NOT RUN against a production adapter**; that adapter does
not exist. Capability probe results do not discharge them.

| Proposed test | Required evidence |
|---|---|
| `test_native_creation_is_private_from_first_handle` | File/directory descriptor at creation, owner TokenUser, protected single ACE; no post-create repair window or inherited handle. |
| `test_native_acl_refuses_unsafe_existing_state` | Wrong owner, null/absent/broad/inherited/unknown ACL and nonowner parent child-deletion rights refuse before bytes/commit. |
| `test_native_identity_and_marker_lineage` | Real file ID/change time/link/ACL fields, schema-3 exact snapshots; POSIX placeholders/schema-2 cannot qualify Windows. |
| `test_native_alias_and_reparse_substitution` | Hardlinks, final symlink, junction/other reparse ancestor, anchor swap and path ABA refuse without protected-state changes. |
| `test_native_content_and_security_mutation` | Same-size content, link-count, owner/DACL and ancestor mutation during/between reads and before commit refuse. |
| `test_native_private_handle_lifecycle` | Injected allocation/query/read/close errors free security descriptors/handles exactly once; no hidden state repair. |
| `test_native_lock_two_processes` | Independent-process contention refuses immediately, wrong-owner lock refuses, normal close/process death releases; stable lock survives pointer replacement. |
| `test_native_copy_publish_is_independent_and_exclusive` | Backup/accepted/reservation bytes and ACL exact; collision and retry adoption never clobber; cleanup returns to one link. |
| `test_native_failure_commit_classification` | File flush, rename, copy cleanup and post-publication namespace failure preserve exits 2/3/4 and retained source/backup/config. |
| `test_native_prepare_and_promote_lineage` | Complete synthetic prepare→capture-fake→promote flow, config-last/pointer-last ordering, frontend authority and acceptance preserved. |
| `test_native_supported_namespace_durability` | Reviewed API/filesystem contract plus separately scheduled crash/fault-injection evidence for backup/accepted/pointer namespace barriers; unsupported filesystems refuse. |
| `test_posix_persistence_regression` | Existing prepare/promotion/enroll synthetic suites on real POSIX, unchanged schema-2/mode-600/flock/durability behavior. |

Any separate-user access test or crash/VM campaign needs explicit resource
scheduling. These unresolved tests give no authority to open retained enrollment,
models, audio, microphone/doctor/live gates or activate a candidate.

## Bounded verification

Windows CPython 3.10.11: 16 admission/refusal tests passed, including fresh import
without `fcntl`, help, synthetic CLI refusal and preservation of a generated
JSON file. The final bounded run also passed 15 existing pure/injected cases
(31 passed, 24 deselected in 1.68 s). The empty-object native capability probe passed protected owner-only
ACL creation/readback, file and directory flush calls, lock contention and release.
This same-process lock probe does not qualify two-process/crash/ABI behavior.

An existing pure fixture fails on unchanged main and this branch:
`test_v5_migrates_v2_v3_v4_only_when_input_agc_was_absent` hardcodes a POSIX
`/m/gtcrn.onnx` descriptor while the implementation normalizes a Windows absolute
path. Its Windows fixture-portability correction is separate from mutation.
No retained private state, hardware, model, doctor, live session or actual
preparation/promotion transaction was opened/run. Commands/results: [WORKLOG](../WORKLOG.md).
