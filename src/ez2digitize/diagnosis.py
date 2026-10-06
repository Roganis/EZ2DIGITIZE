# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plain-language explanations of common backend failures.

A failed step is shown with the end of its log, which means little to most
users. `explain` matches the exit status and the log against known failure
signatures (messages taken from the pinned COLMAP and OpenMVS sources, and
what the system says when a process runs out of memory or disk) and returns
what probably happened and what to try. It returns None for anything it
doesn't recognise; the log tail is still shown.
"""

from __future__ import annotations

import re
import signal
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    title: str
    advice: str

    def __str__(self) -> str:
        return f"{self.title}. {self.advice}"


_OUT_OF_MEMORY = Explanation(
    "The step ran out of memory",
    "Choose a lower detail level, close other programs, or limit the CPU threads "
    "(each thread needs its own memory).",
)
_DISK_FULL = Explanation(
    "The disk is full",
    "Free some space on the drive that holds the project; a reconstruction needs a few GB.",
)
_NO_OVERLAP = Explanation(
    "The cameras could not be placed",
    "The photos probably overlap too little or the object has too little texture. Take "
    "more photos with more overlap between neighbours (each part of the object in at "
    "least three photos), put the object on a patterned mat, or use masks for a turntable.",
)
_UNREADABLE = Explanation(
    "Some photos could not be read",
    "Check the photo checks for unreadable files and leave them out.",
)
_CRASH = Explanation(
    "The tool crashed",
    "This is a bug in the tool or its build, not something you did. Please report it with "
    "the full log (Export diagnostics).",
)
_ILLEGAL_INSTRUCTION = Explanation(
    "The tool needs a newer processor",
    "Its build uses CPU instructions this processor lacks. Please report it with the "
    "full log (Export diagnostics).",
)
_NOTHING_DENSE = Explanation(
    "No dense points could be computed",
    "The camera poses are probably wrong or the views barely overlap. Check how many "
    "photos were placed, and try a lower detail level.",
)
_EMPTY_MESH = Explanation(
    "No mesh could be built",
    "The dense point cloud is empty or too sparse. Try a lower detail level, or check "
    "that the masks don't hide the object.",
)

Rule = Callable[[str, int, str], Explanation | None]


def _killed(exit_code: int, *signals: int | None) -> bool:
    # The runner reports death by signal as a negative exit code; a shell as 128 + n.
    return any(exit_code in (-sig, 128 + sig) for sig in signals if sig is not None)


# POSIX signal numbers; Windows has none of these, its crashes are NTSTATUS codes.
SIGKILL: int | None = getattr(signal, "SIGKILL", None)
SIGBUS: int | None = getattr(signal, "SIGBUS", None)
# Windows: access violation, stack overflow, heap corruption; illegal instruction.
WINDOWS_CRASHES = frozenset({0xC0000005, 0xC00000FD, 0xC0000374, 0xC0000409})
WINDOWS_ILLEGAL = frozenset({0xC000001D})
# Windows: no memory, and the commit limit (RAM plus page file) reached.
WINDOWS_NO_MEMORY = frozenset({0xC0000017, 0xC000012D})


def _windows_status(exit_code: int, codes: frozenset[int]) -> bool:
    # Python reports them unsigned; some tools pass them on as signed 32-bit.
    return (exit_code & 0xFFFFFFFF) in codes


def _matches(pattern: str) -> Callable[[str], bool]:
    regex = re.compile(pattern, re.IGNORECASE)
    return lambda text: regex.search(text) is not None


_oom = _matches(
    r"std::bad_alloc|out of memory|cannot allocate memory|failed to allocate|memory exhausted"
)
_disk = _matches(r"no space left on device|disk quota exceeded")
_no_model = _matches(
    r"failed to create (any )?sparse model|no good initial image pair|global mapping failed"
    r"|cannot continue with empty pose graph|no frames registered|no images with matches"
    r"|cannot continue without image pairs|failed to solve rotation averaging"
)
_unreadable = _matches(r"could not read image|cannot read image at path")
_dense_empty = _matches(
    r"no images see \d+ or more points|no valid depth|densifying point-cloud failed"
    r"|empty point-?cloud|point-cloud is not valid"
)
_mesh_empty = _matches(r"empty initial mesh|empty mesh|cannot load mesh|no faces")


def _rules() -> list[Rule]:
    def memory(stage: str, code: int, log: str) -> Explanation | None:
        # The kernel's OOM killer sends SIGKILL, which also is how cancel ends: the
        # pipeline reports cancellation separately, so a SIGKILL here is not ours.
        if _oom(log) or _killed(code, SIGKILL) or _windows_status(code, WINDOWS_NO_MEMORY):
            return _OUT_OF_MEMORY
        return None

    def disk(stage: str, code: int, log: str) -> Explanation | None:
        return _DISK_FULL if _disk(log) else None

    def illegal(stage: str, code: int, log: str) -> Explanation | None:
        illegal = _killed(code, signal.SIGILL) or _windows_status(code, WINDOWS_ILLEGAL)
        if illegal or "illegal instruction" in log.lower():
            return _ILLEGAL_INSTRUCTION
        return None

    def mapping(stage: str, code: int, log: str) -> Explanation | None:
        return _NO_OVERLAP if stage in ("matching", "mapping") and _no_model(log) else None

    def unreadable(stage: str, code: int, log: str) -> Explanation | None:
        return _UNREADABLE if stage in ("features", "undistort") and _unreadable(log) else None

    def dense(stage: str, code: int, log: str) -> Explanation | None:
        return _NOTHING_DENSE if stage == "densify" and _dense_empty(log) else None

    def mesh(stage: str, code: int, log: str) -> Explanation | None:
        if stage in ("mesh", "refine", "texture") and _mesh_empty(log):
            return _EMPTY_MESH
        return None

    def crash(stage: str, code: int, log: str) -> Explanation | None:
        crashed = _killed(code, signal.SIGSEGV, signal.SIGABRT, SIGBUS) or _windows_status(
            code, WINDOWS_CRASHES
        )
        return _CRASH if crashed or "segmentation fault" in log.lower() else None

    # Most specific first: a crash after "out of memory" is an out-of-memory.
    return [disk, memory, illegal, mapping, unreadable, dense, mesh, crash]


_RULES = _rules()


def explain(stage: str, exit_code: int, log_tail: Sequence[str]) -> Explanation | None:
    """What probably went wrong in a failed `stage`, or None if not recognised."""
    log = "\n".join(log_tail)
    for rule in _RULES:
        found = rule(stage, exit_code, log)
        if found is not None:
            return found
    return None
