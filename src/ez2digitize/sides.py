# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-sided scans: the object photographed, turned over, photographed again.

One capture shows the object standing as it should (the first side), the
other with it turned over (`CaptureBundle.flipped`). Both go into one
reconstruction; what makes that work:

- **Masks for every photo.** Between the two sets the object moved and the
  table didn't, so matching the table (or the background) pulls the sets
  apart. With masks only the object is matched, and it is the same object
  in both. `check` reports photos without a mask.
- **Overlap between the sides.** The sides of the object (its middle) must
  show in both sets: a low ring of photos for each side does it.
- **Up from the first side only.** Exports stand upright from the way the
  photos were held (see orientation); in the turned-over set the object is
  upside down (or on its side), so only the first side's photos count.

After camera placement `joined` says how many photos of each side were
placed together, so the pipeline can tell the user when the sides didn't
connect.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.project import Project


@dataclass(frozen=True)
class Sides:
    """The photos (COLMAP names, `<capture id>/<file>`) of each side."""

    first: list[str]
    turned: list[str]

    @property
    def complete(self) -> bool:
        return bool(self.first) and bool(self.turned)


def sides(bundles: Sequence[CaptureBundle]) -> Sides | None:
    """The photos of each side; None unless some capture is marked as turned over."""
    if not any(b.flipped for b in bundles):
        return None
    return Sides(_names(bundles, flipped=False), _names(bundles, flipped=True))


def _names(bundles: Sequence[CaptureBundle], *, flipped: bool) -> list[str]:
    # As colmap.image_names names them: the photos the reconstruction uses.
    return [
        f"{b.id}/{f.name}"
        for b in bundles
        if b.flipped == flipped
        for f in b.used
        if f.kind == "image"
    ]


def upright_names(bundles: Sequence[CaptureBundle]) -> set[str] | None:
    """The photos whose 'down' is the object's: None means all of them.

    In a two-sided scan, only the first side's; if every capture is marked
    as turned over, there is no first side to go by, and all are used.
    """
    found = sides(bundles)
    if found is None or not found.first:
        return None
    return set(found.first)


def without_mask(project: Project, names: Iterable[str]) -> list[str]:
    """The photos (COLMAP names) that have no mask in use."""
    return [n for n in names if not (project.masks_dir / f"{n}.png").is_file()]


def check(project: Project, bundles: Sequence[CaptureBundle], *, use_masks: bool) -> list[str]:
    """What stands in the way of joining the two sides; empty if nothing does."""
    found = sides(bundles)
    if found is None:
        return []
    problems = []
    if not found.first:
        problems.append(
            "every capture is marked as turned over: import the first side too, or mark the "
            "capture taken with the object standing as the first side"
        )
    if not found.turned:
        problems.append("the turned-over side has no photos yet")
    if not use_masks:
        problems.append("masks are switched off: without them the table pulls the two sides apart")
    else:
        missing = without_mask(project, found.first + found.turned)
        if missing:
            problems.append(
                f"{len(missing)} of {len(found.first) + len(found.turned)} photos have no mask; "
                "both sides need masks for every photo to join (make them in the Masks tab)"
            )
    return problems


@dataclass(frozen=True)
class Joined:
    """How many photos of each side the reconstruction placed together."""

    first: int
    first_total: int
    turned: int
    turned_total: int

    @property
    def ok(self) -> bool:
        return self.first > 0 and self.turned > 0


def joined(found: Sides, placed: Iterable[str]) -> Joined:
    placed_set = set(placed)
    return Joined(
        first=sum(1 for n in found.first if n in placed_set),
        first_total=len(found.first),
        turned=sum(1 for n in found.turned if n in placed_set),
        turned_total=len(found.turned),
    )


def join_notice(result: Joined) -> str:
    """What the pipeline tells the user after camera placement."""
    counts = (
        f"{result.first} of {result.first_total} photos of the first side and "
        f"{result.turned} of {result.turned_total} of the turned-over side"
    )
    if result.ok:
        return f"both sides joined: {counts} placed together"
    return (
        f"the two sides did not join ({counts} placed): the model has one side only. "
        "Check that every photo has a good mask (no table or stand left in), and that "
        "the object's sides show in both sets (a low ring of photos for each side)."
    )
