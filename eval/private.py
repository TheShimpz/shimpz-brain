"""Owner-only files for evaluation transcripts, verdicts, and campaign metadata (ADR-0094).

A transcript holds replies and simulated Action data, so it never lands in a repository and never follows a link:
the destination's directory is reached one component at a time by descriptor with ``O_NOFOLLOW``, must lie outside
every git work tree, and must be owned by the current user and writable by no one else; the file is opened relative
to that descriptor without following a link, must be a regular file with one link owned by the current user, and is
made owner-only before anything is written. This module uses only the standard library.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class PrivateFileError(ValueError):
    """The destination is not a safe owner-only private file."""


def _has_repository_marker(directory: int) -> bool:
    try:
        os.stat(".git", dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _private_directory(parent: Path) -> int:
    """Walk to ``parent`` one component at a time by descriptor, never following a link; refuse a repository.

    Every step opens the next directory relative to the descriptor of the one before with ``O_NOFOLLOW``, so no
    ancestor can be swapped for a link between the check and the use. The final directory must be owned by this user
    and writable by no one else: an admitted private directory.
    """
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in parent.parts[1:]:
            if _has_repository_marker(directory):
                raise PrivateFileError("private files must stay outside every repository")
            try:
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory
                )
            except OSError as exc:
                raise PrivateFileError("private files must not be reached through a symbolic link") from exc
            os.close(directory)
            directory = child
        metadata = os.fstat(directory)
        if _has_repository_marker(directory):
            raise PrivateFileError("private files must stay outside every repository")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise PrivateFileError("private files need a directory owned by this user and writable by no one else")
    except BaseException:
        os.close(directory)
        raise
    return directory


def open_private(path: Path, *, append: bool = False) -> int:
    """Return a write descriptor to an owner-only regular file in an admitted private directory."""
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise PrivateFileError("private files must not be reached through a symbolic link")
    directory = _private_directory(absolute.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_APPEND if append else 0)
    try:
        descriptor = os.open(absolute.name, flags, 0o600, dir_fd=directory)
    except OSError as exc:
        raise PrivateFileError("private file cannot be opened safely") from exc
    finally:
        os.close(directory)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
        os.close(descriptor)
        raise PrivateFileError("private file must be a regular file with one link owned by this user")
    os.fchmod(descriptor, 0o600)
    if not append:
        os.ftruncate(descriptor, 0)
    return descriptor


def write_descriptor(descriptor: int, text: str) -> None:
    """Write to a descriptor ``open_private`` returned and close it."""
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_private(path: Path, text: str) -> None:
    write_descriptor(open_private(path), text)
