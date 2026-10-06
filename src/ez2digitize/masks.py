# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic masks: separate the object from the background in every photo.

The model is ISNet (`isnet-general-use`, Apache-2.0, code and weights), run
by ONNX Runtime on the CPU in a worker process (`mask_worker`). On
a 4-core CI-class CPU it needs about 1 s per photo and 1 GB; BiRefNet, the
other permissively licensed candidate, needed 16-20 s and 7-12 GB, too much
for 8 GB machines. The model (179 MB) is downloaded on first use into the
user's cache and checked against its pinned sha256.

Masks are made per capture, as a stage `masks-<capture id>` whose inputs are
the capture's photos: importing another capture masks only the new one,
leaving a photo out or bringing it back masks nothing again. The stage
writes them under its own folder; `apply_auto` then copies them to the
project's `masks/<capture id>/`, which is what the pipeline reads:

    masks/<capture id>/<image name>.png           masks in use
    masks/<capture id>/dropped/<image name>.png   masks dropped in review
    masks/<capture id>/auto.json                  which of them are automatic

Imported masks are never overwritten by automatic ones. A photo without a
mask keeps everything (see `colmap.extract_features`).
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import re
import shutil
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ez2digitize.backends.colmap import READABLE_SUFFIXES
from ez2digitize.core import download
from ez2digitize.core.capture import CaptureBundle, CaptureError, CaptureFile, import_masks
from ez2digitize.core.download import models_dir as models_dir
from ez2digitize.core.files import (
    FormatError,
    fingerprint,
    read_json_object,
    write_json_atomic,
)
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, EventHandler, Progress
from ez2digitize.core.stage import Backend, StageSpec, load_manifest, run_stage

STAGE_PREFIX = "masks-"
RECORD_FILE = "auto.json"
DROPPED_DIR = "dropped"
JOBS_FILE = "jobs.json"
OUTPUT_DIR = "masks"
REPORT_FILE = "report.json"  # written by the worker, see mask_worker.REPORT_FILE
# Bump when the worker's processing changes, so earlier masks are made again.
MASKER_VERSION = 1

# Flags for the review: a mask the user should look at.
NOTHING_FOUND = 0.005  # coverage below this: the model found no object
EVERYTHING = 0.9  # coverage above this: the background was kept
UNSURE = 0.06  # share of the photo the model was unsure about (skull set: at most 0.047)
UNLIKE_OTHERS = 3.0  # coverage this many times off the capture's median
MIN_FOR_MEDIAN = 5

Flag = Literal["nothing found", "almost the whole photo", "unsure", "unlike the others"]
MaskState = Literal["automatic", "imported", "dropped", "none"]


@dataclass(frozen=True)
class MaskModel:
    name: str
    url: str
    sha256: str
    size: int
    license: str
    input_size: int


MODEL = MaskModel(
    name="isnet-general-use",
    # The ONNX export the rembg project publishes (MIT); the weights are the
    # DIS authors' general-use ISNet, Apache-2.0.
    url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    sha256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    size=178648008,
    license="Apache-2.0",
    input_size=1024,
)


class MaskingError(Exception):
    """Masks can't be made (no model, the worker failed)."""


class MaskingCancelled(MaskingError):
    pass


# --- The model ---------------------------------------------------------------


def model_file(model: MaskModel = MODEL) -> Path:
    return models_dir() / f"{model.name}.onnx"


def find_model(model: MaskModel = MODEL) -> Path | None:
    """The downloaded model, or None. Its hash was checked when it was downloaded."""
    path = model_file(model)
    try:
        return path if path.stat().st_size == model.size else None
    except OSError:
        return None


def download_model(
    model: MaskModel = MODEL,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
    url: str | None = None,
) -> Path:
    """Download the model into `models_dir()`, check its sha256, return its path.

    `on_progress(received, total)` is called as it arrives. A partial
    download is never left under the final name.
    """
    target = model_file(model)
    source = url or model.url
    try:
        return download.fetch(
            source,
            target,
            model.sha256,
            size=model.size,
            on_progress=on_progress,
            cancel=cancel,
        )
    except download.DownloadCancelled as exc:
        raise MaskingCancelled(str(exc)) from exc
    except download.DownloadDamaged as exc:
        raise MaskingError(f"the downloaded model is damaged ({exc})") from exc
    except download.DownloadError as exc:
        raise MaskingError(
            f"could not download the masking model from {source}: {exc.__cause__}. "
            f"Download it by hand and save it as {target}"
        ) from exc


# --- Making masks ------------------------------------------------------------


def worker_argv() -> list[str]:
    """How to start the masking worker: this program, with the worker's entry point."""
    if getattr(sys, "frozen", False):
        # The packaged app's launcher dispatches on this (tools/packaging/entry.py).
        return [sys.executable, "mask-worker"]
    return [sys.executable, "-m", "ez2digitize.mask_worker"]


def stage_name(bundle: CaptureBundle) -> str:
    return f"{STAGE_PREFIX}{bundle.id}"


def maskable(bundle: CaptureBundle) -> list[CaptureFile]:
    """The photos of a bundle that get masks: every one COLMAP can read.

    Left-out photos too, so leaving one out (or bringing it back) doesn't
    make the masks again.
    """
    return [
        f
        for f in bundle.files
        if f.kind == "image" and Path(f.name).suffix.lower() in READABLE_SUFFIXES
    ]


def mask_spec(
    project: Project,
    bundle: CaptureBundle,
    model_path: Path,
    *,
    model: MaskModel = MODEL,
    threads: int = 0,
) -> StageSpec:
    """The stage that masks one capture's photos into `<stage>/masks/<capture id>/`."""
    name = stage_name(bundle)
    stage_dir = project.stage_dir(name)
    files = maskable(bundle)
    if not files:
        raise MaskingError(f"capture {bundle.id} has no photos to mask")

    def prepare(folder: Path) -> None:
        jobs = [
            {
                "key": f.name,
                "image": str(bundle.root / f.name),
                "mask": str(folder / OUTPUT_DIR / bundle.id / f"{f.name}.png"),
            }
            for f in files
        ]
        write_json_atomic(folder / JOBS_FILE, {"jobs": jobs})

    try:
        runtime = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        runtime = "unknown"
    argv: list[str | Path] = [
        *worker_argv(),
        "--model", model_path,
        "--jobs", stage_dir / JOBS_FILE,
        "--size", str(model.input_size),
    ]  # fmt: skip
    if threads:
        argv += ["--threads", str(threads)]
    return StageSpec(
        name=name,
        backend=Backend("onnxruntime", runtime),
        argv=argv,
        parameters={"model": model.name, "masker_version": MASKER_VERSION},
        inputs={
            "model": f"sha256:{model.sha256}",
            "photos": "files:" + fingerprint([(f.name, f.sha256) for f in files]),
        },
        parse_line=MaskProgress(),
        prepare=prepare,
    )


class MaskProgress:
    """Turns the worker's `mask 3/62 IMG_0003.JPG` lines into progress."""

    _LINE = re.compile(r"^mask (\d+)/(\d+) (.+)$")

    def __call__(self, line: str) -> Progress | None:
        if line.startswith("model "):
            return Progress("loading the masking model", 0.0)
        match = self._LINE.match(line.strip())
        if match is None:
            return None
        n, total = int(match[1]), int(match[2])
        return Progress(f"masking {match[3]} ({n}/{total})", (n - 1) / total)


@dataclass
class MaskRun:
    """What `make_masks` did for one capture."""

    capture: str
    reused: bool
    added: int  # automatic masks put in use for the first time
    dropped: int  # of those, dropped straight away (nothing found)


def make_masks(
    project: Project,
    bundles: Sequence[CaptureBundle],
    model_path: Path,
    *,
    on_event: EventHandler | None = None,
    cancel: CancelToken | None = None,
    threads: int = 0,
    force: bool = False,
) -> list[MaskRun]:
    """Mask every bundle's photos (reusing earlier runs) and put the masks in use."""
    runs = []
    for bundle in bundles:
        if not maskable(bundle):
            continue
        spec = mask_spec(project, bundle, model_path, threads=threads)
        run = run_stage(project, spec, on_event=on_event, cancel=cancel, force=force)
        manifest = run.manifest
        if manifest.status == "cancelled":
            raise MaskingCancelled("masking cancelled")
        if not manifest.succeeded:
            log = project.stage_dir(spec.name) / "log.txt"
            raise MaskingError(
                f"masking capture {bundle.id} failed (exit code {manifest.exit_code}); see {log}"
            )
        added, dropped = apply_auto(project, bundle)
        runs.append(MaskRun(bundle.id, run.reused, added, dropped))
    return runs


# --- Masks in use, review ----------------------------------------------------


@dataclass
class _Record:
    """masks/<capture id>/auto.json: the names whose mask is automatic."""

    path: Path
    automatic: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, folder: Path) -> _Record:
        path = folder / RECORD_FILE
        try:
            data = read_json_object(path)
        except FormatError:
            return cls(path)
        names = data.get("automatic", [])
        if not isinstance(names, list):
            return cls(path)
        return cls(path, {n for n in names if isinstance(n, str)})

    def save(self) -> None:
        if self.automatic:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, {"schema_version": 1, "automatic": sorted(self.automatic)})
        else:
            self.path.unlink(missing_ok=True)


def _in_use(project: Project, capture: str, name: str) -> Path:
    return project.masks_dir / capture / f"{name}.png"


def _dropped(project: Project, capture: str, name: str) -> Path:
    return project.masks_dir / capture / DROPPED_DIR / f"{name}.png"


def auto_mask(project: Project, bundle: CaptureBundle, name: str) -> Path | None:
    """The automatic mask the last successful masking run made for a photo."""
    stage_dir = project.stage_dir(stage_name(bundle))
    manifest = load_manifest(stage_dir)
    if manifest is None or not manifest.succeeded:
        return None
    path = stage_dir / OUTPUT_DIR / bundle.id / f"{name}.png"
    return path if path.is_file() else None


def _report(project: Project, bundle: CaptureBundle) -> dict[str, dict[str, float]]:
    stage_dir = project.stage_dir(stage_name(bundle))
    manifest = load_manifest(stage_dir)
    if manifest is None or not manifest.succeeded:
        return {}
    try:
        data = read_json_object(stage_dir / REPORT_FILE)
    except FormatError:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def flags(stats: dict[str, float], median_coverage: float | None) -> list[Flag]:
    """Why a mask is worth a look in the review, from the worker's statistics."""
    coverage = float(stats.get("coverage", 0.0))
    found: list[Flag] = []
    if coverage < NOTHING_FOUND:
        found.append("nothing found")
    elif coverage > EVERYTHING:
        found.append("almost the whole photo")
    elif median_coverage and (
        coverage * UNLIKE_OTHERS < median_coverage or coverage > median_coverage * UNLIKE_OTHERS
    ):
        found.append("unlike the others")
    if coverage >= NOTHING_FOUND and float(stats.get("uncertain", 0.0)) > UNSURE:
        found.append("unsure")
    return found


def _median_coverage(report: dict[str, dict[str, float]]) -> float | None:
    values = [float(s.get("coverage", 0.0)) for s in report.values()]
    return statistics.median(values) if len(values) >= MIN_FOR_MEDIAN else None


def apply_auto(project: Project, bundle: CaptureBundle) -> tuple[int, int]:
    """Put the capture's automatic masks in use; returns (added, dropped of those).

    Imported masks stay. A photo whose automatic mask was dropped stays
    dropped (its mask is refreshed). A new mask where the model found
    nothing would remove the whole photo, so it starts out dropped.
    """
    record = _Record.load(project.masks_dir / bundle.id)
    report = _report(project, bundle)
    median = _median_coverage(report)
    added = dropped = 0
    for entry in maskable(bundle):
        source = auto_mask(project, bundle, entry.name)
        if source is None:
            continue
        in_use = _in_use(project, bundle.id, entry.name)
        dropped_path = _dropped(project, bundle.id, entry.name)
        if entry.name in record.automatic:
            target = dropped_path if dropped_path.exists() else in_use
        elif in_use.exists() or dropped_path.exists():
            continue  # imported
        else:
            added += 1
            empty = "nothing found" in flags(report.get(entry.name, {}), median)
            target = dropped_path if empty else in_use
            dropped += empty
            record.automatic.add(entry.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    record.save()
    return added, dropped


def drop(project: Project, bundle: CaptureBundle, names: Iterable[str]) -> None:
    """Stop using these photos' masks: they are then used whole."""
    for name in names:
        in_use, dropped = _in_use(project, bundle.id, name), _dropped(project, bundle.id, name)
        if in_use.exists():
            dropped.parent.mkdir(parents=True, exist_ok=True)
            in_use.replace(dropped)
        elif not dropped.exists():
            raise CaptureError(f"{bundle.id}/{name} has no mask")


def restore(project: Project, bundle: CaptureBundle, names: Iterable[str]) -> None:
    """Use dropped masks again."""
    for name in names:
        in_use, dropped = _in_use(project, bundle.id, name), _dropped(project, bundle.id, name)
        if dropped.exists():
            dropped.replace(in_use)
        elif not in_use.exists():
            raise CaptureError(f"{bundle.id}/{name} has no mask")


def clear_auto(project: Project, bundle: CaptureBundle) -> int:
    """Remove the capture's automatic masks (in use or dropped); returns how many."""
    folder = project.masks_dir / bundle.id
    record = _Record.load(folder)
    for name in record.automatic:
        _in_use(project, bundle.id, name).unlink(missing_ok=True)
        _dropped(project, bundle.id, name).unlink(missing_ok=True)
    removed = len(record.automatic)
    record.automatic.clear()
    record.save()
    with contextlib.suppress(OSError):  # only if nothing else is left
        (folder / DROPPED_DIR).rmdir()
        folder.rmdir()
    return removed


def import_folder_masks(project: Project, bundle: CaptureBundle, folder: Path) -> list[str]:
    """Import a folder of masks for a capture (see capture.import_masks).

    They replace the automatic or dropped mask of the same photo.
    """
    names = import_masks(project, bundle, folder)
    record = _Record.load(project.masks_dir / bundle.id)
    for name in names:
        record.automatic.discard(name)
        _dropped(project, bundle.id, name).unlink(missing_ok=True)
    record.save()
    return names


@dataclass
class MaskEntry:
    """One photo in the review."""

    capture: str
    name: str
    image: Path
    state: MaskState
    # The mask file: in use, or dropped; None without a mask.
    mask: Path | None
    excluded: bool
    stats: dict[str, Any] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)


def review(project: Project, bundles: Sequence[CaptureBundle]) -> list[MaskEntry]:
    """Every maskable photo with its mask's state and flags, in capture order."""
    entries = []
    for bundle in bundles:
        record = _Record.load(project.masks_dir / bundle.id)
        report = _report(project, bundle)
        median = _median_coverage(report)
        for f in maskable(bundle):
            in_use = _in_use(project, bundle.id, f.name)
            dropped = _dropped(project, bundle.id, f.name)
            state: MaskState
            if in_use.exists():
                state, mask = ("automatic" if f.name in record.automatic else "imported"), in_use
            elif dropped.exists():
                state, mask = "dropped", dropped
            else:
                state, mask = "none", None
            stats = report.get(f.name, {}) if f.name in record.automatic else {}
            entries.append(
                MaskEntry(
                    capture=bundle.id,
                    name=f.name,
                    image=bundle.root / f.name,
                    state=state,
                    mask=mask,
                    excluded=f.excluded,
                    stats=stats,
                    flags=flags(stats, median) if stats else [],
                )
            )
    return entries


# How to turn a photo stored with an EXIF orientation upright (PIL's table).
UPRIGHT = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
ORIENTATION_TAG = 0x0112
TINT = (220, 30, 30)


def preview(image: Path, mask: Path | None, size: tuple[int, int] = (160, 120)) -> Image.Image:
    """A thumbnail of the photo, upright, within `size`, with what the mask removes red."""
    with Image.open(image) as img:
        orientation = int(img.getexif().get(ORIENTATION_TAG, 1))
        img.draft("RGB", (size[0] * 2, size[0] * 2))
        thumb = img.convert("RGB")
    # Fit the upright picture: sideways photos are fitted with width and height swapped.
    sideways = orientation in (5, 6, 7, 8)
    thumb.thumbnail((size[1], size[0]) if sideways else size)
    if mask is not None:
        with Image.open(mask) as m:
            kept = m.convert("L").resize(thumb.size, Image.Resampling.BILINEAR)
        tinted = Image.blend(thumb, Image.new("RGB", thumb.size, TINT), 0.6)
        thumb = Image.composite(thumb, tinted, kept)
    if orientation in UPRIGHT:
        thumb = thumb.transpose(UPRIGHT[orientation])
    return thumb


def has_masks(project: Project) -> bool:
    """True if any photo has a mask in use (dropped ones don't count)."""
    return project.masks_dir.is_dir() and any(project.masks_dir.glob("*/*.png"))
