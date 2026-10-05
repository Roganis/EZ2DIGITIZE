# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Photo checks: what in a capture is likely to hurt the reconstruction.

Each photo is inspected once: pixel size, camera EXIF (make, model, lens,
focal length) and a sharpness score. The result is kept in the photo's entry
in capture.json, as `metadata["photo"]`, next to the original file it
describes. The checks then only compare those numbers, so they are cheap
and run whenever a project is shown.

Findings are advice: nothing is refused or removed. The sharpness score is
the variance of the Laplacian of the photo scaled to 1024 px. It depends
on the subject, so photos are only compared with the rest of their set.
"""

from __future__ import annotations

import statistics
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import ExifTags, Image, ImageFilter, ImageStat

from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.resources import cpu_threads

PHOTO_KEY = "photo"
# Bump when inspection changes; older results are then inspected again.
INSPECT_VERSION = 1
SHARPNESS_SIZE = 1024

# A photo this much less sharp than the median of its set is flagged. On a
# 12 MP turntable set the softest good photo scored 0.39; a 4 px blur 0.36.
BLUR_RATIO = 0.35
BLUR_MIN_SET = 5
MIN_SHORT_SIDE = 1000
MIN_PHOTOS = 20
# Focal lengths this close count as one zoom setting: EXIF rounds to whole
# millimetres, and a nudged zoom ring is absorbed by calibration (a skull
# set at 39 and 40 mm reconstructs fine).
FOCAL_TOLERANCE = 0.05
INSPECT_THREADS = 4

Level = Literal["error", "warning"]


@dataclass(frozen=True)
class PhotoInfo:
    width: int = 0
    height: int = 0
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    focal_mm: float | None = None
    focal_35mm: float | None = None
    sharpness: float | None = None
    # Why the file can't be read as an image; the other fields are then empty.
    error: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def body(self) -> tuple[str | None, str | None, str | None]:
        """Camera body and lens."""
        return self.make, self.model, self.lens

    def to_dict(self) -> dict[str, Any]:
        return {"version": INSPECT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Any) -> PhotoInfo | None:
        """None for missing, outdated or malformed results."""
        if not isinstance(data, dict) or data.get("version") != INSPECT_VERSION:
            return None
        fields = {k: v for k, v in data.items() if k != "version"}
        try:
            return cls(**fields)
        except TypeError:
            return None


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    message: str
    # Bundle the finding is about, None for the whole project.
    capture: str | None = None
    files: tuple[str, ...] = ()


# --- inspection ------------------------------------------------------------------


def inspect_photo(path: Path) -> PhotoInfo:
    """Size, camera EXIF and sharpness of one photo (never raises for bad files)."""
    try:
        with warnings.catch_warnings():
            # Very large photos (medium format, stitched panoramas) are legitimate.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                exif = image.getexif()
                details = exif.get_ifd(ExifTags.IFD.Exif)
                # JPEG decodes straight to a fraction of its size; about 8x faster.
                image.draft("L", (SHARPNESS_SIZE, SHARPNESS_SIZE))
                gray = image.convert("L")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return PhotoInfo(error=str(exc) or type(exc).__name__)
    return PhotoInfo(
        width=width,
        height=height,
        make=_text(exif.get(ExifTags.Base.Make)),
        model=_text(exif.get(ExifTags.Base.Model)),
        lens=_text(details.get(ExifTags.Base.LensModel)),
        focal_mm=_number(details.get(ExifTags.Base.FocalLength)),
        focal_35mm=_number(details.get(ExifTags.Base.FocalLengthIn35mmFilm)),
        sharpness=_sharpness(gray),
    )


def measure_sharpness(path: Path) -> float | None:
    """Sharpness score of an image file alone (None if unreadable)."""
    try:
        with Image.open(path) as image:
            image.draft("L", (SHARPNESS_SIZE, SHARPNESS_SIZE))
            gray = image.convert("L")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return _sharpness(gray)


def _sharpness(gray: Image.Image) -> float:
    """Variance of the Laplacian of the image scaled to SHARPNESS_SIZE."""
    gray.thumbnail((SHARPNESS_SIZE, SHARPNESS_SIZE))
    laplacian = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128))
    return round(ImageStat.Stat(laplacian).var[0], 3)


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    # Cameras pad these fields with spaces or NULs.
    text = value.replace("\x00", "").strip()
    return text or None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(number, 3) if number > 0 else None


def photo_infos(bundle: CaptureBundle) -> dict[str, PhotoInfo | None]:
    """Stored inspection result of each photo in use, None where it is missing."""
    return {
        f.name: PhotoInfo.from_dict(f.metadata.get(PHOTO_KEY))
        for f in bundle.used
        if f.kind == "image"
    }


def needs_inspection(bundle: CaptureBundle) -> bool:
    return any(info is None for info in photo_infos(bundle).values())


def inspect_bundle(
    bundle: CaptureBundle,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    threads: int | None = None,
) -> int:
    """Inspect the photos that have no current result and save capture.json.

    Excluded photos are inspected too, so bringing one back needs no wait.
    Returns how many were inspected. `on_progress(done, total)` is called
    after each photo, from the calling thread.
    """
    todo = [
        f
        for f in bundle.files
        if f.kind == "image" and PhotoInfo.from_dict(f.metadata.get(PHOTO_KEY)) is None
    ]
    if not todo:
        return 0
    workers = threads or min(INSPECT_THREADS, cpu_threads())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(inspect_photo, [bundle.root / f.name for f in todo])
        for done, (entry, info) in enumerate(zip(todo, results, strict=True), start=1):
            entry.metadata[PHOTO_KEY] = info.to_dict()
            if on_progress is not None:
                on_progress(done, len(todo))
    bundle.save()
    return len(todo)


# --- checks ----------------------------------------------------------------------


def check_project(bundles: Sequence[CaptureBundle]) -> list[Finding]:
    """Findings for every capture and the project as a whole.

    Excluded photos and photos not inspected yet (see `inspect_bundle`) are
    left out.
    """
    findings: list[Finding] = []
    readable = 0
    for bundle in bundles:
        infos = {name: info for name, info in photo_infos(bundle).items() if info is not None}
        findings += check_capture(bundle.id, infos, source=bundle.source)
        readable += sum(1 for info in infos.values() if info.error is None)
    findings += _duplicates(bundles)
    if bundles and readable < MIN_PHOTOS and not any(needs_inspection(b) for b in bundles):
        findings.append(
            Finding(
                "warning",
                "few-photos",
                f"Only {readable} photos. Aim for 30 or more, taken all around the object "
                "with plenty of overlap between neighbouring shots.",
            )
        )
    return findings


def check_capture(
    capture: str, infos: Mapping[str, PhotoInfo], *, source: str = "folder"
) -> list[Finding]:
    """Findings for one capture: its photos are expected to come from one camera.

    Video frames never have a focal length in EXIF, so its absence isn't
    reported for video captures.
    """
    findings: list[Finding] = []

    def add(level: Level, code: str, message: str, files: Iterable[str] = ()) -> None:
        findings.append(Finding(level, code, message, capture, tuple(sorted(files))))

    unreadable = [name for name, info in infos.items() if info.error is not None]
    if unreadable:
        add(
            "error",
            "unreadable",
            f"{_count(unreadable, 'file')} can't be read as an image and will not be used.",
            unreadable,
        )
    photos = {name: info for name, info in infos.items() if info.error is None}
    if not photos:
        return findings

    sizes = Counter(info.size for info in photos.values())
    (common_size, common_count), *_ = sizes.most_common()
    odd = [name for name, info in photos.items() if info.size != common_size]
    if odd and common_count >= 2 * len(odd):
        width, height = common_size
        add(
            "warning",
            "odd-size",
            f"{_count(odd, 'photo')} {_are(odd)} a different size from the others "
            f"({width}×{height}): another camera, a crop or a screenshot? Leave "
            f"{'it' if len(odd) == 1 else 'them'} out if not part of this capture.",
            odd,
        )
        main = {name: info for name, info in photos.items() if info.size == common_size}
    else:
        main = photos

    small = [name for name, info in main.items() if min(info.size) < MIN_SHORT_SIDE]
    if small:
        add(
            "warning",
            "low-resolution",
            f"{_count(small, 'photo')} {_are(small)} under {MIN_SHORT_SIDE} pixels on the "
            "short side, so fine detail will be lost.",
            small,
        )

    no_focal = [name for name, info in main.items() if info.focal_mm is None]
    if source == "video":
        no_focal = []
    elif len(no_focal) == len(main):
        add(
            "warning",
            "no-focal-length",
            "The photos have no focal length in their EXIF data, so it has to be estimated, "
            "which needs more photos and can fail. Keep EXIF data when copying or exporting "
            "photos.",
        )
    elif no_focal:
        add(
            "warning",
            "no-focal-length",
            f"{_count(no_focal, 'photo')} {_have(no_focal)} no focal length in their EXIF "
            "data, unlike the rest: edited or from another source?",
            no_focal,
        )

    # Photos without a focal length are already reported above.
    cameras = _cameras(
        i for n, i in main.items() if n not in no_focal or len(no_focal) == len(main)
    )
    if len(cameras) > 1:
        described = "; ".join(_describe_camera(camera) for camera in cameras[:3])
        add(
            "warning",
            "mixed-cameras",
            f"These photos come from {len(cameras)} cameras or zoom settings ({described}), "
            "which are calibrated as one. Import the photos of each camera and zoom setting "
            "separately.",
        )

    scores = {name: info.sharpness for name, info in main.items() if info.sharpness}
    if len(scores) >= BLUR_MIN_SET:
        median = statistics.median(scores.values())
        blurry = [name for name, score in scores.items() if score < BLUR_RATIO * median]
        if blurry:
            add(
                "warning",
                "blurry",
                f"{_count(blurry, 'photo')} {_are(blurry)} much less sharp than the rest "
                "(camera shake or missed focus). Blurry photos add little and can add errors.",
                blurry,
            )
    return findings


def _duplicates(bundles: Sequence[CaptureBundle]) -> list[Finding]:
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for bundle in bundles:
        for entry in bundle.used:
            if entry.kind == "image":
                by_hash[entry.sha256].append((bundle.id, entry.name))
    findings = []
    for copies in by_hash.values():
        if len(copies) > 1:
            first_capture = copies[0][0]
            capture = first_capture if all(c == first_capture for c, _ in copies) else None
            names = tuple(f"{c}/{n}" if capture is None else n for c, n in copies)
            findings.append(
                Finding(
                    "warning",
                    "duplicate",
                    f"{names[0]} was imported {len(copies)} times; the copies add nothing.",
                    capture,
                    names,
                )
            )
    return findings


Camera = tuple[str | None, str | None, str | None, float | None]


def _cameras(infos: Iterable[PhotoInfo]) -> list[Camera]:
    """The distinct cameras (body, lens, zoom setting), most used first.

    Focal lengths within FOCAL_TOLERANCE of a group's shortest one join it.
    """
    focals: dict[tuple[str | None, str | None, str | None], Counter[float | None]] = defaultdict(
        Counter
    )
    for info in infos:
        focals[info.body][info.focal_mm] += 1
    groups: list[tuple[int, Camera]] = []
    for body, counts in focals.items():
        if None in counts:
            groups.append((counts[None], (*body, None)))
        start, size = 0.0, 0
        for focal in sorted(f for f in counts if f is not None):
            if size and focal <= start * (1 + FOCAL_TOLERANCE):
                size += counts[focal]
                continue
            if size:
                groups.append((size, (*body, start)))
            start, size = focal, counts[focal]
        if size:
            groups.append((size, (*body, start)))
    return [camera for _, camera in sorted(groups, key=lambda g: -g[0])]


def _describe_camera(camera: Camera) -> str:
    make, model, lens, focal = camera
    name = model or make or "unknown camera"
    if make and model and not model.lower().startswith(make.lower()):
        name = f"{make} {model}"
    if lens:
        name += f", {lens}"
    if focal:
        name += f" at {focal:g} mm"
    return name


def _count(names: Sequence[str], noun: str) -> str:
    if len(names) == 1:
        return names[0]
    return f"{len(names)} {noun}s"


def _are(names: Sequence[str]) -> str:
    return "is" if len(names) == 1 else "are"


def _have(names: Sequence[str]) -> str:
    return "has" if len(names) == 1 else "have"
