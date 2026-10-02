"""Owner-only files for evaluation transcripts, verdicts, and campaign metadata (ADR-0094).

A transcript holds replies and simulated Action data, so it never lands in a repository and never follows a link:
the destination must be outside every git work tree, its directory must not be reached through a symbolic link, the
file itself is opened without following one, must be a regular file with one link owned by the current user, and is
made owner-only before anything is written. This module uses only the standard library.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class PrivateFileError(ValueError):
    """The destination is not a safe owner-only private file."""


def _inside_repository(directory: Path) -> bool:
    return any((candidate / ".git").exists() for candidate in (directory, *directory.parents))


def open_private(path: Path, *, append: bool = False) -> int:
    """Return a write descriptor to an owner-only regular file outside every repository."""
    absolute = path.absolute()
    if ".." in absolute.parts or absolute.parent.resolve() != absolute.parent:
        raise PrivateFileError("private files must not be reached through a symbolic link")
    if _inside_repository(absolute.parent):
        raise PrivateFileError("private files must stay outside every repository")
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_APPEND if append else 0)
    try:
        descriptor = os.open(absolute, flags, 0o600)
    except OSError as exc:
        raise PrivateFileError("private file cannot be opened safely") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
        os.close(descriptor)
        raise PrivateFileError("private file must be a regular file with one link owned by this user")
    os.fchmod(descriptor, 0o600)
    if not append:
        os.ftruncate(descriptor, 0)
    return descriptor


def write_private(path: Path, text: str) -> None:
    with os.fdopen(open_private(path), "w", encoding="utf-8") as handle:
        handle.write(text)
