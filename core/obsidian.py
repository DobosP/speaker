"""Bounded, read-only access to a configured Obsidian Markdown vault.

The capability deliberately exposes one small surface: find relevant Markdown
notes (or an exact vault-relative note) and return bounded excerpts.  It never
writes the vault, follows symlinks, scans Obsidian/git internals, or returns an
absolute host path.  Note text is PRIVATE and spotlight-fenced as untrusted
file data before it reaches any planner or synthesis model.
"""
from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Iterator, Mapping, Optional

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.origin import Origin
from always_on_agent.text import keywords, normalize_text
from always_on_agent.untrusted import wrap_untrusted

log = logging.getLogger("speaker.obsidian")

_CAPABILITY = "vault.search"
_DEFAULT_ROOT = "~/work/dobo-brain/paul-brain"
_SKIP_DIRS = frozenset({".git", ".obsidian"})
_SUMMARY_RE = re.compile(r"(?m)^summary\s*:\s*(.*?)\s*$")
_TITLE_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
_QUERY_NOISE = frozenset(
    {
        "brain",
        "about",
        "check",
        "could",
        "dobo",
        "do",
        "find",
        "look",
        "lookup",
        "my",
        "note",
        "notes",
        "obsidian",
        "read",
        "say",
        "search",
        "second",
        "show",
        "summarize",
        "tell",
        "vault",
    }
)
_MAX_QUERY_CHARS = 512
_MAX_QUERY_BYTES = 2048
_SAFE_DESCRIPTOR_WALK = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in getattr(os, "supports_dir_fd", ())
    and os.stat in getattr(os, "supports_dir_fd", ())
    and os.stat in getattr(os, "supports_follow_symlinks", ())
    and os.scandir in getattr(os, "supports_fd", ())
)


def _bounded_int(value: object, default: int, *, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(low, min(high, parsed))


def _explicit_true(value: object) -> bool:
    """Default-deny JSON boolean parsing for the opt-in feature gate."""

    return value is True


@dataclass(frozen=True)
class ObsidianConfig:
    """Machine-local vault path plus hard-clamped read/output budgets."""

    enabled: bool = False
    vault_root: str = _DEFAULT_ROOT
    max_files: int = 1000
    max_entries: int = 10000
    max_directories: int = 1000
    max_total_bytes: int = 4 * 1024 * 1024
    max_file_bytes: int = 128 * 1024
    max_results: int = 4
    max_excerpt_chars: int = 600
    max_output_chars: int = 3600

    @classmethod
    def from_dict(
        cls, data: Optional[Mapping[str, object]]
    ) -> "ObsidianConfig":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            enabled=_explicit_true(data.get("enabled", False)),
            vault_root=str(data.get("vault_root", _DEFAULT_ROOT) or _DEFAULT_ROOT),
            max_files=_bounded_int(
                data.get("max_files"), 1000, low=1, high=2000
            ),
            max_entries=_bounded_int(
                data.get("max_entries"), 10000, low=1, high=20000
            ),
            max_directories=_bounded_int(
                data.get("max_directories"), 1000, low=1, high=2000
            ),
            max_total_bytes=_bounded_int(
                data.get("max_total_bytes"),
                4 * 1024 * 1024,
                low=1024,
                high=8 * 1024 * 1024,
            ),
            max_file_bytes=_bounded_int(
                data.get("max_file_bytes"),
                128 * 1024,
                low=512,
                high=256 * 1024,
            ),
            max_results=_bounded_int(
                data.get("max_results"), 4, low=1, high=8
            ),
            max_excerpt_chars=_bounded_int(
                data.get("max_excerpt_chars"), 600, low=80, high=1000
            ),
            max_output_chars=_bounded_int(
                data.get("max_output_chars"), 3600, low=800, high=6000
            ),
        )


@dataclass(frozen=True)
class _Match:
    path: str
    title: str
    summary: str
    excerpt: str
    score: int


@dataclass
class _ScanState:
    entries: int = 0
    directories: int = 1  # the configured root
    files: int = 0
    bytes: int = 0
    truncated: bool = False
    entry_limit_reached: bool = False
    io_errors: int = 0
    root_unavailable: bool = False
    unsafe_entries: int = 0


def _cancelled(cancel: object) -> bool:
    return isinstance(cancel, Event) and cancel.is_set()


def _frontmatter_body(text: str) -> tuple[str, str]:
    """Return ``(frontmatter, body)`` without depending on a YAML package."""

    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end < 0:
        return "", text
    return text[4:end], text[end + 5 :]


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    text = _one_line(text)
    if len(text) <= limit:
        return text
    marker = " …[truncated]"
    return text[: max(0, limit - len(marker))].rstrip() + marker


def _clip_output(text: str, limit: int) -> str:
    """Bound content independently of spotlight-envelope configuration."""

    if len(text) <= limit:
        return text
    marker = "\n…[output truncated]"
    return text[: max(0, limit - len(marker))].rstrip() + marker


class ObsidianVault:
    """A bounded reader with stable ordering for fully enumerated scans."""

    def __init__(self, config: ObsidianConfig):
        if not _SAFE_DESCRIPTOR_WALK:
            raise OSError("safe no-follow vault traversal is unavailable")
        expanded = Path(os.path.expanduser(config.vault_root))
        root = expanded.resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError("configured vault root is not a directory")
        self.config = config
        self.root = root
        self._root_fd = -1
        expected = root.stat(follow_symlinks=False)
        probe = self._open_root_path(root)
        try:
            actual = os.fstat(probe)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise OSError("configured vault root changed during setup")
            # A readable-but-non-searchable directory (for example mode 0400)
            # can be listed yet cannot open children. Prove both operations on
            # the retained descriptor before making an availability claim.
            traversal_probe = os.open(
                ".",
                self._directory_flags(),
                dir_fd=probe,
            )
            os.close(traversal_probe)
            listing_probe = os.dup(probe)
            try:
                with os.scandir(listing_probe) as iterator:
                    next(iterator, None)
            finally:
                os.close(listing_probe)
        except Exception:
            os.close(probe)
            raise
        self._root_fd = probe

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )

    @classmethod
    def _open_root_path(cls, root: Path) -> int:
        """Open every absolute path component with no-follow semantics."""

        current = os.open("/", cls._directory_flags())
        try:
            for part in root.parts[1:]:
                following = os.open(
                    part,
                    cls._directory_flags(),
                    dir_fd=current,
                )
                os.close(current)
                current = following
            return current
        except OSError:
            os.close(current)
            raise

    def _open_root(self) -> int:
        root_fd = getattr(self, "_root_fd", -1)
        if root_fd < 0:
            raise OSError("vault reader is closed")
        return os.dup(root_fd)

    def close(self) -> None:
        """Release the retained root descriptor; safe to call repeatedly."""

        root_fd = getattr(self, "_root_fd", -1)
        if root_fd >= 0:
            self._root_fd = -1
            try:
                os.close(root_fd)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _safe_component(name: str) -> bool:
        return bool(name) and name not in (".", "..") and name.isprintable()

    def _open_directory(self, root_fd: int, parts: tuple[str, ...]) -> int:
        """Open a relative directory chain without following any component."""

        current = os.dup(root_fd)
        try:
            for part in parts:
                following = os.open(
                    part,
                    self._directory_flags(),
                    dir_fd=current,
                )
                os.close(current)
                current = following
            return current
        except OSError:
            os.close(current)
            raise

    def _entry_names(
        self,
        directory_fd: int,
        cancel: object,
        state: _ScanState,
    ) -> list[str]:
        """Read at most the remaining global entry budget, polling cancellation."""

        names: list[str] = []
        try:
            with os.scandir(directory_fd) as iterator:
                for entry in iterator:
                    if _cancelled(cancel):
                        return []
                    if state.entries >= self.config.max_entries:
                        state.truncated = True
                        state.entry_limit_reached = True
                        break
                    state.entries += 1
                    names.append(entry.name)
        except OSError:
            state.io_errors += 1
            state.truncated = True
            return []
        return sorted(names, key=lambda name: (name.casefold(), name))

    def _markdown_files(
        self,
        cancel: object,
        state: _ScanState,
    ) -> Iterator[tuple[str, int]]:
        """Yield opened regular Markdown files from a bounded no-follow walk.

        Each yielded descriptor is opened relative to the already-open parent
        directory and verified with ``fstat``. The descriptor—not a pathname—is
        then read, closing file and parent symlink-swap races.
        """

        try:
            root_fd = self._open_root()
        except OSError:
            state.io_errors += 1
            state.truncated = True
            state.root_unavailable = True
            return
        stack: list[tuple[str, ...]] = [()]
        try:
            while stack:
                if _cancelled(cancel):
                    return
                parts = stack.pop()
                try:
                    directory_fd = self._open_directory(root_fd, parts)
                except OSError:
                    state.io_errors += 1
                    state.truncated = True
                    continue
                try:
                    names = self._entry_names(directory_fd, cancel, state)
                    if _cancelled(cancel):
                        return
                    subdirs: list[tuple[str, ...]] = []
                    for name in names:
                        if _cancelled(cancel):
                            return
                        if name in _SKIP_DIRS:
                            continue
                        if not self._safe_component(name):
                            state.unsafe_entries += 1
                            state.truncated = True
                            continue
                        try:
                            child_fd = os.open(
                                name,
                                self._directory_flags(),
                                dir_fd=directory_fd,
                            )
                        except OSError:
                            child_fd = -1
                            try:
                                child_info = os.stat(
                                    name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                            except OSError:
                                state.io_errors += 1
                                state.truncated = True
                                continue
                            if stat.S_ISLNK(child_info.st_mode):
                                continue
                            if stat.S_ISDIR(child_info.st_mode):
                                state.io_errors += 1
                                state.truncated = True
                                continue
                        if child_fd >= 0:
                            os.close(child_fd)
                            if state.directories >= self.config.max_directories:
                                state.truncated = True
                            else:
                                state.directories += 1
                                subdirs.append((*parts, name))
                            continue
                        if not name.lower().endswith(".md"):
                            continue
                        if (
                            state.files >= self.config.max_files
                            or state.bytes >= self.config.max_total_bytes
                        ):
                            state.truncated = True
                            return
                        flags = (
                            os.O_RDONLY
                            | os.O_NOFOLLOW
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NONBLOCK", 0)
                        )
                        file_fd = -1
                        try:
                            file_fd = os.open(name, flags, dir_fd=directory_fd)
                            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                                os.close(file_fd)
                                continue
                        except OSError:
                            if file_fd >= 0:
                                os.close(file_fd)
                            try:
                                info = os.stat(
                                    name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                            except OSError:
                                state.io_errors += 1
                                state.truncated = True
                            else:
                                if not stat.S_ISLNK(info.st_mode):
                                    state.io_errors += 1
                                    state.truncated = True
                            continue
                        relative = "/".join((*parts, name))
                        try:
                            yield relative, file_fd
                        finally:
                            os.close(file_fd)
                        if (
                            state.files >= self.config.max_files
                            or state.bytes >= self.config.max_total_bytes
                        ):
                            state.truncated = True
                            return
                    if not state.entry_limit_reached:
                        stack.extend(reversed(subdirs))
                finally:
                    os.close(directory_fd)
                if state.entry_limit_reached:
                    return
        finally:
            os.close(root_fd)

    def _read_bounded(
        self, file_fd: int, remaining: int
    ) -> tuple[str, int, bool, bool]:
        budget = max(0, min(self.config.max_file_bytes, remaining))
        if budget <= 0:
            return "", 0, True, False
        chunks: list[bytes] = []
        consumed = 0
        try:
            size = os.fstat(file_fd).st_size
        except OSError:
            return "", 0, False, True
        try:
            while consumed < budget:
                chunk = os.read(file_fd, min(64 * 1024, budget - consumed))
                if not chunk:
                    break
                chunks.append(chunk)
                consumed += len(chunk)
        except OSError:
            raw = b"".join(chunks)
            return raw.decode("utf-8", errors="replace"), consumed, True, True
        raw = b"".join(chunks)
        truncated = size > budget
        return raw.decode("utf-8", errors="replace"), len(raw), truncated, False

    @staticmethod
    def _terms(query: str) -> tuple[str, ...]:
        useful = [word for word in keywords(query, limit=16) if word not in _QUERY_NOISE]
        if useful:
            return tuple(useful)
        return ()

    @staticmethod
    def _score(
        *, query: str, terms: tuple[str, ...], path: str, title: str, summary: str, body: str
    ) -> int:
        norm_query = normalize_text(query)
        fields = {
            "path": normalize_text(path),
            "title": normalize_text(title),
            "summary": normalize_text(summary),
            "body": normalize_text(body),
        }
        score = 0
        if norm_query:
            score += 80 if fields["path"] == norm_query else 0
            score += 50 if fields["title"] == norm_query else 0
            score += 20 if norm_query in fields["summary"] else 0
            score += 8 if norm_query in fields["body"] else 0
        if not terms:
            return max(1, score)
        weights = {"path": 12, "title": 10, "summary": 6, "body": 2}
        matched = 0
        for term in terms:
            term_hit = False
            for name, value in fields.items():
                if term in value.split():
                    score += weights[name]
                    term_hit = True
                elif term in value:
                    score += max(1, weights[name] // 2)
                    term_hit = True
            matched += int(term_hit)
        if terms and not matched:
            return 0
        return score + matched * 3

    def _excerpt(self, body: str, query: str, terms: tuple[str, ...]) -> str:
        clean = _one_line(body)
        if not clean:
            return ""
        normalized = normalize_text(clean)
        needles = [normalize_text(query), *terms]
        position = -1
        for needle in needles:
            if not needle:
                continue
            position = normalized.find(needle)
            if position >= 0:
                break
        # ``normalize_text`` is close enough in length for an excerpt anchor; the
        # bound, not character-perfect highlighting, is the security contract.
        if position < 0:
            position = 0
        half = self.config.max_excerpt_chars // 2
        start = max(0, position - half)
        end = min(len(clean), start + self.config.max_excerpt_chars)
        if end - start < self.config.max_excerpt_chars:
            start = max(0, end - self.config.max_excerpt_chars)
        excerpt = clean[start:end].strip()
        if start:
            excerpt = "… " + excerpt
        if end < len(clean):
            excerpt += " …"
        return excerpt

    def search(self, query: str, *, cancel: object = None) -> CapabilityResult:
        raw_query = str(query or "").strip()
        base_data: dict[str, object] = {
            "sensitivity": "private",
            "egress": False,
            "origin": Origin.FILE.value,
            "results": [],
        }
        if not raw_query:
            return CapabilityResult(False, "", data=base_data, error="empty vault query")
        if len(raw_query) > _MAX_QUERY_CHARS or len(raw_query.encode("utf-8")) > _MAX_QUERY_BYTES:
            return CapabilityResult(False, "", data=base_data, error="vault query is too long")
        if _cancelled(cancel):
            return CapabilityResult(True, "", data={**base_data, "cancelled": True})

        terms = self._terms(raw_query)
        matches: list[_Match] = []
        state = _ScanState()
        for relative, file_fd in self._markdown_files(cancel, state):
            if _cancelled(cancel):
                return CapabilityResult(
                    True,
                    "",
                    data={
                        **base_data,
                        "cancelled": True,
                        "scanned_files": state.files,
                        "scanned_bytes": state.bytes,
                        "scanned_entries": state.entries,
                        "scanned_directories": state.directories,
                    },
                )
            remaining = self.config.max_total_bytes - state.bytes
            text, consumed, file_truncated, read_error = self._read_bounded(
                file_fd, remaining
            )
            state.files += 1
            state.bytes += consumed
            state.truncated = state.truncated or file_truncated
            if read_error:
                state.io_errors += 1
                state.truncated = True
            if _cancelled(cancel):
                return CapabilityResult(
                    True,
                    "",
                    data={
                        **base_data,
                        "cancelled": True,
                        "scanned_files": state.files,
                        "scanned_bytes": state.bytes,
                        "scanned_entries": state.entries,
                        "scanned_directories": state.directories,
                    },
                )
            if not text:
                continue
            frontmatter, body = _frontmatter_body(text)
            summary_match = _SUMMARY_RE.search(frontmatter)
            title_match = _TITLE_RE.search(body)
            title = (
                _one_line(title_match.group(1))
                if title_match
                else Path(relative).stem
            )
            summary = _one_line(summary_match.group(1)).strip("'\"") if summary_match else ""
            score = self._score(
                query=raw_query,
                terms=terms,
                path=relative,
                title=title,
                summary=summary,
                body=body,
            )
            if score <= 0:
                continue
            matches.append(
                _Match(
                    path=relative,
                    title=title,
                    summary=_clip(summary, 300),
                    excerpt=self._excerpt(body, raw_query, terms),
                    score=score,
                )
            )

        if _cancelled(cancel):
            return CapabilityResult(
                True,
                "",
                data={
                    **base_data,
                    "cancelled": True,
                    "scanned_files": state.files,
                    "scanned_bytes": state.bytes,
                    "scanned_entries": state.entries,
                    "scanned_directories": state.directories,
                },
            )
        if state.root_unavailable:
            return CapabilityResult(
                False,
                "",
                data={
                    **base_data,
                    "scanned_files": state.files,
                    "scanned_bytes": state.bytes,
                    "scanned_entries": state.entries,
                    "scanned_directories": state.directories,
                    "io_errors": state.io_errors,
                    "unsafe_entries": state.unsafe_entries,
                },
                error="vault unavailable",
            )
        matches.sort(
            key=lambda match: (-match.score, match.path.casefold(), match.path)
        )
        selected = matches[: self.config.max_results]
        rendered: list[str] = []
        result_rows: list[dict[str, object]] = []
        for match in selected:
            lead = f"{match.path} — {match.title}"
            if match.summary:
                lead += f". Summary: {match.summary}"
            if match.excerpt:
                lead += f". Excerpt: {match.excerpt}"
            rendered.append(lead)
            result_rows.append({"path": match.path, "score": match.score})

        omitted_matches = len(matches) > len(selected)
        if state.truncated:
            qualifier = (
                "Some vault entries had unsafe names and were skipped; results may be incomplete."
                if state.unsafe_entries
                else (
                    "Some vault content was unreadable; results may be incomplete."
                    if state.io_errors
                    else "Vault scan limits were reached; results may be incomplete."
                )
            )
            plain = f"{qualifier}\n" + (
                "\n".join(rendered)
                if rendered
                else "No match was found within the configured scan limits."
            )
        elif omitted_matches:
            plain = "Additional matching notes were omitted by the result limit.\n" + (
                "\n".join(rendered)
            )
        else:
            plain = (
                "\n".join(rendered)
                if rendered
                else "No matching Markdown notes were found in the configured vault."
            )
        plain = _clip_output(plain, self.config.max_output_chars)
        fenced = wrap_untrusted(
            plain,
            source="vault",
            max_chars=self.config.max_output_chars,
        )
        data = {
            **base_data,
            "results": result_rows,
            "scanned_files": state.files,
            "scanned_bytes": state.bytes,
            "scanned_entries": state.entries,
            "scanned_directories": state.directories,
            "io_errors": state.io_errors,
            "unsafe_entries": state.unsafe_entries,
            "truncated": state.truncated or omitted_matches,
        }
        return CapabilityResult(
            True,
            fenced,
            data=data,
            citations=tuple(f"vault:{match.path}" for match in selected),
        )


def attach_obsidian_capability(
    registry: CapabilityRegistry,
    config: ObsidianConfig,
    *,
    vault: Optional[ObsidianVault] = None,
) -> CapabilityRegistry:
    """Register ``vault.search`` only when its configured root is usable."""

    if not config.enabled:
        return registry
    try:
        reader = vault or ObsidianVault(config)
    except (OSError, RuntimeError, ValueError):
        # Truthful availability: an absent/malformed machine-local path does not
        # create a planner tool or a user-facing capability claim.
        log.info("vault.search unavailable: configured vault root is not readable")
        return registry

    def provider(query: str, context: dict[str, object]) -> CapabilityResult:
        return reader.search(query, cancel=context.get("cancel_event"))

    registry.register(
        _CAPABILITY,
        provider,
        spec=CapabilitySpec(
            name=_CAPABILITY,
            summary="search and read bounded excerpts from your local Obsidian notes",
            when_to_use=(
                "search the configured local vault only when the user explicitly "
                "refers to their own notes, my vault, my Obsidian, or the configured "
                "Paul/dobo brain; use web search for generic Obsidian topics"
            ),
            egress="local",
            speaks=True,
            side_effecting=False,
            planner_tool=True,
            user_facing=True,
        ),
    )
    log.info("vault.search registered (read-only, bounded, private)")
    return registry


__all__ = [
    "ObsidianConfig",
    "ObsidianVault",
    "attach_obsidian_capability",
]
