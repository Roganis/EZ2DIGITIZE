# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP: features, matching, sparse mapping and undistortion, on the CPU.

Option names follow the pinned version (4.2.1), which renamed many of them
(`SiftExtraction.use_gpu` became `FeatureExtraction.use_gpu`, ...). Other
versions are reported as unsupported; their commands may fail.

Stage layout. Images are read straight from the capture bundles: COLMAP's
image path is the project's `captures/` folder and every image is named by
its path relative to it (`<capture id>/IMG_0001.jpg`), listed in an image
list. Masks follow COLMAP's own naming, `<masks>/<capture id>/IMG_0001.jpg.png`.

    features/   database.db, image_list.txt, masks/ (a mask for every image)
    matching/   database.db (a copy of the features one, then matched)
    mapping/    sparse/0, sparse/1, ... (one folder per model)
    undistort/  images/, sparse/ (pinhole model, input for OpenMVS and Brush)
    mask-undistort/  masks/<image stem>.mask.png (masks warped like undistort/)

Matching works on a copy of the database so the features stage's output is
never modified and both stay valid for caching.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import sqlite3
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image

from ez2digitize.backends.colmap_model import read_cameras, read_image_cameras
from ez2digitize.backends.common import (
    BackendError,
    BackendMissing,
    find_tool,
    result_parameters,
)
from ez2digitize.core import download
from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.files import fingerprint, write_uniform_png
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, ProcessStartError, Progress, run_quick
from ez2digitize.core.stage import (
    Backend,
    StageManifest,
    StageSpec,
    capture_input,
    load_manifest,
    stage_input,
    tree_input,
)

NAME = "colmap"
PINNED_VERSION = "4.2.1"
ENV_VAR = "EZ2D_COLMAP"
DATABASE = "database.db"
IMAGE_LIST = "image_list.txt"
# Formats COLMAP reads directly; HEIC and others need converting first.
READABLE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

CameraGrouping = Literal["per_capture", "per_image", "single"]
MatchMode = Literal["exhaustive", "sequential", "spatial", "vocab_tree"]
# SIFT (COLMAP's classic, on the CPU), or learned: ALIKED features matched
# with LightGlue, run with ONNX Runtime; better on weak texture and large
# viewpoint changes, slower on the CPU, and the models are downloaded once.
FeatureKind = Literal["sift", "aliked"]
MapperKind = Literal["incremental", "global"]


@dataclass(frozen=True)
class Colmap:
    path: Path
    version: str

    @property
    def backend(self) -> Backend:
        return Backend(NAME, self.version)

    @property
    def supported(self) -> bool:
        return self.version == PINNED_VERSION


def parse_version(text: str) -> str | None:
    """Version from `colmap help` output ("COLMAP 4.2.1 (Commit ...)")."""
    match = re.search(r"COLMAP (\d+\.\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def locate(explicit: Path | None = None) -> Colmap:
    path = find_tool("colmap", explicit=explicit, env_var=ENV_VAR)
    if path is None:
        where = str(explicit) if explicit else f"${ENV_VAR}, the app bundle or PATH"
        raise BackendMissing(f"COLMAP not found ({where})")
    try:
        text = run_quick([path, "help"])
    except ProcessStartError as exc:
        raise BackendMissing(f"COLMAP at {path} can't be run: {exc}") from exc
    version = parse_version(text)
    if version is None:
        raise BackendMissing(f"{path} doesn't look like COLMAP (no version in 'colmap help')")
    return Colmap(path=path, version=version)


# --- files COLMAP would download -------------------------------------------------


@dataclass(frozen=True)
class Pinned:
    """A file COLMAP 4.2.1 itself downloads, as its source pins it (URL and sha256).

    Our builds have downloads off, so the app fetches these into the user's
    cache when a run needs them, like the masking model.
    """

    name: str
    url: str
    sha256: str
    size: int  # bytes, for progress and to say how much is fetched

    @property
    def path(self) -> Path:
        return download.models_dir() / self.name


_RELEASES = "https://github.com/colmap/colmap/releases/download"
# src/colmap/retrieval/resources.h (kDefault*VocabTreeUri): image retrieval,
# one tree per kind of feature.
VOCAB_TREES: dict[FeatureKind, Pinned] = {
    "sift": Pinned(
        "vocab_tree_faiss_flickr100K_words256K.bin",
        f"{_RELEASES}/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
        "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc",
        72_412_636,
    ),
    "aliked": Pinned(
        "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
        f"{_RELEASES}/3.13.0/vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
        "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e",
        18_764_565,
    ),
}
# src/colmap/feature/resources.h: ALIKED (BSD-3-Clause, Zhao et al.) and the
# LightGlue weights for it (Apache-2.0, Lindenberger et al.), as ONNX models.
ALIKED_MODEL = Pinned(
    "aliked-n16rot.onnx",
    f"{_RELEASES}/3.13.0/aliked-n16rot.onnx",
    "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    2_997_054,
)
LIGHTGLUE_MODEL = Pinned(
    "aliked-lightglue.onnx",
    f"{_RELEASES}/3.13.0/aliked-lightglue.onnx",
    "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    45_804_950,
)


def find_pinned(pinned: Pinned) -> Path | None:
    """The downloaded file, or None (its hash was checked when it arrived)."""
    path = pinned.path
    try:
        return path if path.stat().st_size == pinned.size else None
    except OSError:
        return None


def fetch_pinned(
    pinned: Pinned,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    """Download it into the user's cache; raises core.download.DownloadError."""
    return download.fetch(
        pinned.url,
        pinned.path,
        pinned.sha256,
        size=pinned.size,
        on_progress=on_progress,
        cancel=cancel,
    )


# --- options -------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureOptions:
    max_image_size: int = 3200
    max_num_features: int = 8192
    camera_model: str = "SIMPLE_RADIAL"
    # One set of intrinsics per capture bundle by default: a bundle is one
    # session with one camera. Phones that switch lenses need "per_image".
    camera_grouping: CameraGrouping = "per_capture"
    kind: FeatureKind = "sift"
    # ALIKED: the model file (ALIKED_MODEL). It keeps a quarter of
    # max_num_features (2048 at 8192, COLMAP's own default for ALIKED): its
    # features are more distinctive, and LightGlue's time grows with them.
    model: Path | None = None
    threads: int | None = None


@dataclass(frozen=True)
class MatchOptions:
    """How photos are paired for matching.

    - exhaustive: every pair; best, but grows with the square of the count.
    - sequential: each photo with the next `sequential_overlap` (and, with
      COLMAP's quadratic overlap, the 2nd, 4th, 8th... after), in name order:
      video frames, or a walk through a room. With a vocabulary tree it also
      looks for loops (back at the start of an orbit or a walk).
    - spatial: each photo with its nearest neighbours by EXIF GPS position
      (outdoors, phones).
    - vocab_tree: each photo with the `vocab_tree_images` most similar ones,
      by image retrieval: large unordered sets.

    `vocab_tree` is the tree file (see `VOCAB_TREE`), for vocab_tree and for
    loop detection in sequential matching.
    """

    mode: MatchMode = "exhaustive"
    sequential_overlap: int = 10
    spatial_neighbors: int = 50
    spatial_max_distance_m: float = 100.0
    vocab_tree_images: int = 100
    vocab_tree: Path | None = None
    # The features' kind; ALIKED features are matched with LightGlue
    # (`lightglue`, the LIGHTGLUE_MODEL file).
    features: FeatureKind = "sift"
    lightglue: Path | None = None
    threads: int | None = None


@dataclass(frozen=True)
class MapperOptions:
    # Global SfM by default: on a real turntable capture (62 photos, 39 mm on
    # APS-C) the incremental mapper split into 9 partial models with focal
    # lengths 20-60% off, while the global mapper placed all 62 photos in one
    # model (0.50 px reprojection error) in a fifth of the time.
    kind: MapperKind = "global"
    threads: int | None = None


@dataclass(frozen=True)
class UndistortOptions:
    # Larger images are scaled down; it bounds OpenMVS's memory use.
    max_image_size: int = 3200


# Peak memory per feature-extraction thread at max_image_size 3200: COLMAP
# 4.2.1's CPU SIFT, measured on 12 MP photos (7.8 GB with 4 threads). It
# scales with the image area. COLMAP defaults to one thread per core, which
# on a 16-core desktop asks for 30 GB or more.
FEATURE_MEMORY_PER_THREAD = 2.0 * 1024**3
MEMORY_SHARE = 0.75  # of the available memory, leaving room for everything else


def feature_threads(max_image_size: int, available: int, cpus: int) -> int:
    """Threads for feature extraction that fit in `available` bytes of memory."""
    per_thread = FEATURE_MEMORY_PER_THREAD * (max_image_size / 3200) ** 2
    return max(1, min(cpus, int(available * MEMORY_SHARE // per_thread)))


# --- stages --------------------------------------------------------------------


def image_names(bundles: Sequence[CaptureBundle]) -> list[str]:
    """COLMAP image names (`<capture id>/<file>`) of the images `bundles` use."""
    names = []
    unreadable = []
    for bundle in bundles:
        for f in bundle.used:
            if f.kind != "image":
                continue
            if Path(f.name).suffix.lower() not in READABLE_SUFFIXES:
                unreadable.append(f"{bundle.id}/{f.name}")
            names.append(f"{bundle.id}/{f.name}")
    if unreadable:
        raise BackendError(
            f"COLMAP can't read {len(unreadable)} image(s) without conversion, e.g. {unreadable[0]}"
        )
    if not names:
        raise BackendError("no images to reconstruct from")
    return names


def extract_features(
    colmap: Colmap,
    project: Project,
    bundles: Sequence[CaptureBundle],
    *,
    stage: str = "features",
    masks: Path | None = None,
    options: FeatureOptions | None = None,
) -> StageSpec:
    """`feature_extractor` over the images of `bundles`, optionally masked."""
    options = options or FeatureOptions()
    stage_dir = project.stage_dir(stage)
    names = image_names(bundles)
    argv: list[str | Path] = [
        colmap.path, "feature_extractor",
        "--database_path", stage_dir / DATABASE,
        "--image_path", project.captures_dir,
        "--image_list_path", stage_dir / IMAGE_LIST,
        "--ImageReader.camera_model", options.camera_model,
        "--ImageReader.single_camera", _flag(options.camera_grouping == "single"),
        "--ImageReader.single_camera_per_folder", _flag(options.camera_grouping == "per_capture"),
        # Without any of these COLMAP shares a camera per EXIF model, ignoring
        # lens, zoom and size; "per_image" really means one per image.
        "--ImageReader.single_camera_per_image", _flag(options.camera_grouping == "per_image"),
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.max_image_size", str(options.max_image_size),
    ]  # fmt: skip
    if options.kind == "aliked":
        if options.model is None:
            raise BackendError("ALIKED features need the ALIKED model file")
        argv += [
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.n16rot_model_path", options.model,
            "--AlikedExtraction.max_num_features", str(aliked_features(options)),
        ]  # fmt: skip
    else:
        argv += ["--SiftExtraction.max_num_features", str(options.max_num_features)]
    if options.threads:
        argv += ["--FeatureExtraction.num_threads", str(options.threads)]
    inputs = {"captures": fingerprint([capture_input(b) for b in bundles])}
    if masks is not None:
        argv += ["--ImageReader.mask_path", stage_dir / MASKS_OUT]
        inputs["masks"] = tree_input(masks)

    def prepare(folder: Path) -> None:
        (folder / IMAGE_LIST).write_text("\n".join(names) + "\n", encoding="utf-8")
        if masks is not None:
            stage_masks(project, masks, names, folder / MASKS_OUT)

    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=argv,
        parameters={
            **result_parameters(options),
            # The model by what it is, not where it is.
            "model": ALIKED_MODEL.sha256 if options.kind == "aliked" else None,
            "masked": masks is not None,
        },
        inputs=inputs,
        parse_line=ColmapProgress(total_images=len(names)),
        prepare=prepare,
    )


def aliked_features(options: FeatureOptions) -> int:
    return max(options.max_num_features // 4, 256)


def stage_masks(project: Project, masks: Path, names: Sequence[str], folder: Path) -> None:
    """A mask for every image: the project's, or a white one (keep everything).

    COLMAP skips an image whose mask is missing, so photos without a mask
    (none imported, or dropped in review) get a white mask of their size.
    """
    for name in names:
        target = folder / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        source = masks / f"{name}.png"
        if source.is_file():
            _link_or_copy(source, target)
        else:
            with Image.open(project.captures_dir / name) as image:
                width, height = image.size  # the raw layout, as COLMAP reads it
            write_uniform_png(target, width, height, 255)


def merge_cameras(database: Path, groups: Mapping[str, str]) -> int:
    """Give images of the same group one shared camera; returns cameras removed.

    For captures that mix cameras or zoom settings: features are extracted
    with a camera per image, then every group (image name -> group label)
    is merged into its first camera here. COLMAP 4.2.1 gives each camera a
    trivial rig and each image a frame of that rig; the result matches what
    `single_camera` extraction writes: one camera, one rig, frames of it.
    Images not in `groups` keep their own camera.
    """
    removed = 0
    with contextlib.closing(sqlite3.connect(database)) as db, db:
        images = db.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
        keep: dict[str, int] = {}
        moves: dict[int, int] = {}  # camera -> camera it merges into
        for _image_id, name, camera_id in images.fetchall():
            group = groups.get(name)
            if group is None:
                continue
            target = keep.setdefault(group, camera_id)
            if camera_id != target:
                moves[camera_id] = target
        rig_of = dict(
            db.execute("SELECT ref_sensor_id, rig_id FROM rigs WHERE ref_sensor_type = 0")
        )
        for old, new in moves.items():
            db.execute("UPDATE images SET camera_id = ? WHERE camera_id = ?", (new, old))
            db.execute(
                "UPDATE frame_data SET sensor_id = ? WHERE sensor_id = ? AND sensor_type = 0",
                (new, old),
            )
            if old in rig_of and new in rig_of:
                db.execute(
                    "UPDATE frames SET rig_id = ? WHERE rig_id = ?", (rig_of[new], rig_of[old])
                )
                db.execute("DELETE FROM rig_sensors WHERE rig_id = ?", (rig_of[old],))
                db.execute("DELETE FROM rigs WHERE rig_id = ?", (rig_of[old],))
            db.execute("DELETE FROM cameras WHERE camera_id = ?", (old,))
            removed += 1
    return removed


def match_features(
    colmap: Colmap,
    project: Project,
    features: StageManifest,
    *,
    stage: str = "matching",
    options: MatchOptions | None = None,
    camera_groups: Mapping[str, str] | None = None,
) -> StageSpec:
    """Feature matching over the pairs `options.mode` chooses (see MatchOptions).

    `camera_groups` (image name -> group, see `merge_cameras`) merges the
    per-image cameras of features extracted with `camera_grouping="per_image"`
    in this stage's copy of the database, before matching.
    """
    options = options or MatchOptions()
    stage_dir = project.stage_dir(stage)
    source_db = project.stage_dir(features.stage) / DATABASE
    argv: list[str | Path] = [
        colmap.path, f"{options.mode}_matcher",
        "--database_path", stage_dir / DATABASE,
        "--FeatureMatching.use_gpu", "0",
    ]  # fmt: skip
    tree = options.vocab_tree
    if options.features == "aliked":
        if options.lightglue is None:
            raise BackendError("ALIKED features are matched with LightGlue, which needs its model")
        argv += [
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--AlikedMatching.lightglue_model_path", options.lightglue,
        ]  # fmt: skip
    if options.mode == "sequential":
        argv += [
            "--SequentialMatching.overlap", str(options.sequential_overlap),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.loop_detection", _flag(tree is not None),
        ]  # fmt: skip
        if tree is not None:
            argv += ["--SequentialMatching.vocab_tree_path", tree]
    elif options.mode == "spatial":
        # COLMAP reads the positions from EXIF GPS during feature extraction.
        argv += [
            "--SpatialMatching.max_num_neighbors", str(options.spatial_neighbors),
            "--SpatialMatching.max_distance", str(options.spatial_max_distance_m),
            "--SpatialMatching.ignore_z", "1",
        ]  # fmt: skip
    elif options.mode == "vocab_tree":
        if tree is None:
            raise BackendError("vocabulary tree matching needs the tree file")
        argv += [
            "--VocabTreeMatching.vocab_tree_path", tree,
            "--VocabTreeMatching.num_images", str(options.vocab_tree_images),
        ]  # fmt: skip
    if options.threads:
        argv += ["--FeatureMatching.num_threads", str(options.threads)]

    def prepare(folder: Path) -> None:
        shutil.copy2(source_db, folder / DATABASE)
        if camera_groups:
            merge_cameras(folder / DATABASE, camera_groups)

    # The tree and the model by what they are, not where they are.
    parameters = {
        **result_parameters(options),
        "vocab_tree": VOCAB_TREES[options.features].sha256 if tree else None,
        "lightglue": LIGHTGLUE_MODEL.sha256 if options.features == "aliked" else None,
    }
    if camera_groups:
        parameters["camera_groups"] = fingerprint(sorted(camera_groups.items()))
    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=argv,
        parameters=parameters,
        inputs={"features": stage_input(features)},
        parse_line=ColmapProgress(),
        prepare=prepare,
    )


PAIRS = "pairs.txt"
# Pairs from known poses: each photo with this many nearest others (by
# camera position) among those looking within MAX_PAIR_ANGLE_DEG of it.
PAIR_NEIGHBOURS = 20
MAX_PAIR_ANGLE_DEG = 60.0

Vector = tuple[float, float, float]


def nearby_pairs(
    cameras: Mapping[str, tuple[Vector, Vector]],
    *,
    neighbours: int = PAIR_NEIGHBOURS,
    max_angle_deg: float = MAX_PAIR_ANGLE_DEG,
) -> list[tuple[str, str]]:
    """The photo pairs worth matching, from known poses: (name, name), each once.

    `cameras`: image name -> (camera centre, viewing direction), all in one
    world. Each photo is paired with its nearest neighbours among those
    looking the same way; only distances are compared, so the world's unit
    doesn't matter.
    """
    names = sorted(cameras)
    cos_limit = math.cos(math.radians(max_angle_deg))
    views = {name: _unit(cameras[name][1]) for name in names}
    pairs: set[tuple[str, str]] = set()
    for a in names:
        centre = cameras[a][0]
        candidates = [
            b
            for b in names
            if b != a and sum(x * y for x, y in zip(views[a], views[b], strict=True)) >= cos_limit
        ]
        candidates.sort(key=lambda b: math.dist(centre, cameras[b][0]))
        for b in candidates[:neighbours]:
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def _unit(v: Vector) -> Vector:
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def match_listed(
    colmap: Colmap,
    project: Project,
    features: StageManifest,
    pairs: Sequence[tuple[str, str]],
    *,
    stage: str = "matching",
    mode: str = "pairs",
    options: MatchOptions | None = None,
    camera_groups: Mapping[str, str] | None = None,
) -> StageSpec:
    """Feature matching of exactly these pairs (`matches_importer`).

    `mode` names where the pairs came from in the manifest's parameters;
    `options.mode` is not used.
    """
    if not pairs:
        raise BackendError("no photo pairs to match")
    options = options or MatchOptions()
    stage_dir = project.stage_dir(stage)
    spec = match_features(
        colmap, project, features, stage=stage, options=options, camera_groups=camera_groups
    )
    argv: list[str | Path] = [
        colmap.path, "matches_importer",
        "--database_path", stage_dir / DATABASE,
        "--match_list_path", stage_dir / PAIRS,
        "--match_type", "pairs",
        "--FeatureMatching.use_gpu", "0",
    ]  # fmt: skip
    if options.features == "aliked":
        argv += [
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--AlikedMatching.lightglue_model_path", options.lightglue or "",
        ]  # fmt: skip
    if options.threads:
        argv += ["--FeatureMatching.num_threads", str(options.threads)]
    prepare_database = spec.prepare
    lines = "".join(f"{a} {b}\n" for a, b in pairs)

    def prepare(folder: Path) -> None:
        if prepare_database is not None:
            prepare_database(folder)
        (folder / PAIRS).write_text(lines, encoding="utf-8")

    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=argv,
        parameters={**spec.parameters, "mode": mode, "pairs": fingerprint(list(pairs))},
        inputs=spec.inputs,
        parse_line=ColmapProgress(),
        prepare=prepare,
    )


def map_sparse(
    colmap: Colmap,
    project: Project,
    matching: StageManifest,
    *,
    total_images: int | None = None,
    stage: str = "mapping",
    options: MapperOptions | None = None,
) -> StageSpec:
    """Incremental `mapper` or `global_mapper`; models land in `<stage>/sparse/<n>`."""
    options = options or MapperOptions()
    stage_dir = project.stage_dir(stage)
    command = "mapper" if options.kind == "incremental" else "global_mapper"
    argv: list[str | Path] = [
        colmap.path, command,
        "--database_path", project.stage_dir(matching.stage) / DATABASE,
        "--image_path", project.captures_dir,
        "--output_path", stage_dir / "sparse",
    ]  # fmt: skip
    if options.threads:
        prefix = "Mapper" if options.kind == "incremental" else "GlobalMapper"
        argv += [f"--{prefix}.num_threads", str(options.threads)]
    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=argv,
        parameters=result_parameters(options),
        inputs={"matching": stage_input(matching)},
        parse_line=ColmapProgress(total_images=total_images),
        prepare=lambda folder: (folder / "sparse").mkdir(),
    )


def undistort(
    colmap: Colmap,
    project: Project,
    mapping: StageManifest,
    *,
    model: Path,
    stage: str = "undistort",
    options: UndistortOptions | None = None,
) -> StageSpec:
    """`image_undistorter` for one model of the mapping stage (see `best_model`)."""
    options = options or UndistortOptions()
    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=[
            colmap.path,
            "image_undistorter",
            "--image_path",
            project.captures_dir,
            "--input_path",
            model,
            "--output_path",
            project.stage_dir(stage),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(options.max_image_size),
        ],  # fmt: skip
        parameters={**result_parameters(options), "model": model.name},
        inputs={"mapping": stage_input(mapping)},
        parse_line=ColmapProgress(),
    )


MASK_INPUT = "mask_cameras.txt"
MASKS_OUT = "masks"
MASK_SUFFIX = ".mask.png"  # OpenMVS's naming: <image stem>.mask.png


def undistort_masks(
    colmap: Colmap,
    project: Project,
    mapping: StageManifest,
    *,
    model: Path,
    masks: Path,
    stage: str = "mask-undistort",
    options: UndistortOptions | None = None,
) -> StageSpec:
    """Masks warped like the images of `undistort`, named the way OpenMVS wants them.

    `image_undistorter` only undistorts the photos. This stage runs
    `image_undistorter_standalone` on the masks with each image's camera and
    the same size limit, which is the same computation, so the masks line up
    with the undistorted photos. Output: `<stage>/masks/<image stem>.mask.png`
    for every registered image, OpenMVS's `--mask-path` naming. Images without
    a mask get a white one (all kept), since OpenMVS needs one per image.

    Masks are `<masks>/<image name>.png`, COLMAP's naming (see
    `extract_features`). Use the same `options` as for `undistort`.
    """
    options = options or UndistortOptions()
    stage_dir = project.stage_dir(stage)
    cameras = read_cameras(model)
    image_cameras = read_image_cameras(model)
    _check_unique_stems(image_cameras)

    def prepare(folder: Path) -> None:
        staged = folder / "input"
        staged.mkdir()
        lines = []
        for name, camera_id in sorted(image_cameras.items()):
            camera = cameras[camera_id]
            target = staged / f"{PurePosixPath(name).stem}{MASK_SUFFIX}"
            source = masks / f"{name}.png"
            if source.is_file():
                _link_or_copy(source, target)
            else:
                write_uniform_png(target, camera.width, camera.height, 255)
            lines.append(f"{target.name} {camera.to_text()}")
        (folder / MASK_INPUT).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / MASKS_OUT).mkdir()

    return StageSpec(
        name=stage,
        backend=colmap.backend,
        argv=[
            colmap.path,
            "image_undistorter_standalone",
            "--image_path",
            stage_dir / "input",
            "--input_file",
            stage_dir / MASK_INPUT,
            "--output_path",
            stage_dir / MASKS_OUT,
            "--max_image_size",
            str(options.max_image_size),
        ],  # fmt: skip
        parameters={**result_parameters(options), "model": model.name},
        inputs={"mapping": stage_input(mapping), "masks": tree_input(masks)},
        parse_line=ColmapProgress(),
        prepare=prepare,
    )


def _check_unique_stems(image_cameras: dict[str, int]) -> None:
    # OpenMVS finds a mask by the image's file stem alone, in one folder.
    seen: dict[str, str] = {}
    for name in sorted(image_cameras):
        stem = PurePosixPath(name).stem
        if stem in seen:
            raise BackendError(
                f"{seen[stem]} and {name} have the same file name; OpenMVS can't tell "
                f"their masks apart. Rename one of them, or run without masks."
            )
        seen[stem] = name


def _link_or_copy(source: Path, target: Path) -> None:
    # A hard link costs no space; fall back to copying across file systems.
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


# --- results -------------------------------------------------------------------


def matched_database(project: Project) -> Path | None:
    """The matching stage's database, if the current camera placement came from it.

    Not when a plugin placed the cameras: the database left from an earlier
    COLMAP run would describe other matches.
    """
    matching = load_manifest(project.stage_dir("matching"))
    mapping = load_manifest(project.stage_dir("mapping"))
    if matching is None or mapping is None or not matching.succeeded:
        return None
    if mapping.inputs.get("matching") != f"run:{matching.run_id}":
        return None
    database = project.stage_dir("matching") / DATABASE
    return database if database.is_file() else None


def registered_images(model: Path) -> int:
    """Number of registered images in a binary model (first field of images.bin)."""
    try:
        with (model / "images.bin").open("rb") as fh:
            header = fh.read(8)
    except OSError:
        return 0
    return struct.unpack("<Q", header)[0] if len(header) == 8 else 0


def models(sparse_dir: Path) -> list[Path]:
    """Model folders written by the mapper, in COLMAP's order."""
    if not sparse_dir.is_dir():
        return []
    found = [d for d in sparse_dir.iterdir() if d.is_dir() and (d / "images.bin").is_file()]
    return sorted(found, key=lambda d: (len(d.name), d.name))


def best_model(sparse_dir: Path) -> Path | None:
    """The model with the most registered images; disconnected sets make several."""
    candidates = models(sparse_dir)
    return max(candidates, key=registered_images) if candidates else None


# --- progress ------------------------------------------------------------------

# glog: severity, date (yyyymmdd or mmdd), time, thread id (decimal, or hex on macOS),
# file:line.
_GLOG_PREFIX = re.compile(r"^[IWEF]\d{4,8} [\d:.]+\s+\S+ [\w.+-]+:\d+\] ")
_COUNTED = re.compile(
    r"^(Processed file|Processing file|Processing image|Processing batch|Undistorting image)"
    r" \[(\d+)/(\d+)\]"
)
_BLOCK = re.compile(r"^Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
_REGISTERING = re.compile(r"^Registering image #\d+ \((?:num_reg_frames=)?(\d+)\)")
_LABELS = {
    "Processed file": "Extracting features",
    "Processing file": "Extracting features",
    "Processing image": "Matching",
    "Processing batch": "Matching",
    "Undistorting image": "Undistorting images",
}


class ColmapProgress:
    """Turns COLMAP's log lines into progress events.

    Section headings (the line after a row of `=`) become messages, counted
    lines (`[i/n]`) become fractions. The mapper only says how many images
    are registered so far, so its fraction needs the total number of images.
    """

    def __init__(self, total_images: int | None = None) -> None:
        self.total_images = total_images
        # Headings are framed by two rules: "" -> "heading" -> "closing" -> "".
        self._heading_state = ""

    def __call__(self, line: str) -> Progress | None:
        text = _GLOG_PREFIX.sub("", line).strip()
        if text and set(text) == {"="}:
            self._heading_state = "heading" if self._heading_state != "closing" else ""
            return None
        if self._heading_state == "heading":
            self._heading_state = "closing"
            if text:
                return Progress(text)
        if m := _COUNTED.match(text):
            done, total = int(m.group(2)), int(m.group(3))
            label = _LABELS[m.group(1)]
            return Progress(f"{label} {done}/{total}", done / total if total else None)
        if m := _BLOCK.match(text):
            row, n, col, n2 = (int(g) for g in m.groups())
            fraction = ((row - 1) * n2 + col) / (n * n2) if n and n2 else None
            return Progress(f"Matching block {row}/{n}, {col}/{n2}", fraction)
        if m := _REGISTERING.match(text):
            registered = int(m.group(1))
            if self.total_images:
                fraction = min(registered / self.total_images, 1.0)
                return Progress(f"Registered {registered}/{self.total_images} images", fraction)
            return Progress(f"Registered {registered} images")
        return None


def _flag(value: bool) -> str:
    return "1" if value else "0"
