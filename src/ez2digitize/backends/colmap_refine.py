# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Refine a camera placement plugin's poses with COLMAP.

Feed-forward networks (VGGT, MapAnything) place the cameras from images a
few hundred pixels across, with no bundle adjustment: good enough to know
which photos see what, not for the dense cloud's sub-pixel needs. With
refinement the plugin's model is only the start:

1. features as usual (`colmap.extract_features`);
2. matching of the pairs the plugin's poses say overlap (`pose_pairs`):
   each photo with its nearest neighbours looking the same way, instead of
   every pair or an image-retrieval guess (`match_pairs`,
   `matches_importer`);
3. points triangulated from those matches at full resolution with the
   poses held (`triangulate`, `point_triangulator`);
4. photos that got almost no points left out (`drop_weak`,
   `image_filterer`): their poses were too far off to triangulate with;
5. COLMAP's mapper continuing from that model (`refine`): it places the
   left-out photos afresh, then refines poses, intrinsics and points
   together, re-triangulating and filtering as it goes. This is the stage
   the rest of the pipeline reads as the camera placement.

On the synthetic scene (32 photos, a textured box), poses 2° off with a
focal length 7% too long came out 0.06° off (median) with the focal length
within 0.1%; at 5° and 15%, 0.06° median and 0.9° at worst.

`point_triangulator` (COLMAP 4.2.1, `src/colmap/exe/sfm.cc`) matches the
model's images to the database by name, but then loads the database's
cameras, rigs and frames into the model and stops if a camera with the
same id has another model or size, or a rig or frame differs. A plugin's
model has its own cameras (one PINHOLE per photo, say), so the input model
is rebuilt from the database itself (`write_known_poses_model`): its
cameras, rigs, frames and image ids, with the plugin's poses and, as the
starting intrinsics, the plugin's focal length and principal point (the
database only has a guess from EXIF, and the triangulation holds them).
"""

from __future__ import annotations

import contextlib
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ez2digitize.backends import colmap
from ez2digitize.backends.colmap_model import CAMERA_MODELS, Camera, read_cameras, read_images
from ez2digitize.backends.common import BackendError
from ez2digitize.core.project import Project
from ez2digitize.core.stage import StageManifest, StageSpec, stage_input

# Photos seeing fewer points than this after triangulation are placed afresh.
MIN_POINTS = 25
KNOWN_POSES = "known_poses"
CAMERA_SENSOR = 0  # COLMAP's SensorType::CAMERA

Vector = tuple[float, float, float]


# --- pairs ---------------------------------------------------------------------------


def pose_pairs(
    model_dir: Path,
    *,
    neighbours: int = colmap.PAIR_NEIGHBOURS,
    max_angle_deg: float = colmap.MAX_PAIR_ANGLE_DEG,
) -> list[tuple[str, str]]:
    """The photo pairs worth matching, from a model's poses (`colmap.nearby_pairs`)."""
    cameras = {
        name: (pose.centre, pose.rotation[2])  # rotation[2]: the camera's z axis in the world
        for name, pose in read_images(model_dir).items()
    }
    return colmap.nearby_pairs(cameras, neighbours=neighbours, max_angle_deg=max_angle_deg)


# --- the known-poses model -----------------------------------------------------------

# Models whose parameters start f, cx, cy (one focal length), and those that
# start fx, fy, cx, cy; the rest of the parameters are distortion.
ONE_FOCAL = frozenset(
    {
        "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE", "SIMPLE_DIVISION", "DIVISION", "SIMPLE_FISHEYE",
    }
)  # fmt: skip
MODEL_IDS = {name: model_id for model_id, (name, _count) in CAMERA_MODELS.items()}


def intrinsics(camera: Camera) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy of a camera of any COLMAP model."""
    p = camera.params
    if camera.model in ONE_FOCAL:
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def params_for(model: str, fx: float, fy: float, cx: float, cy: float) -> list[float]:
    """Parameters of `model` with these intrinsics and no distortion."""
    count = CAMERA_MODELS[MODEL_IDS[model]][1]
    values = [(fx + fy) / 2, cx, cy] if model in ONE_FOCAL else [fx, fy, cx, cy]
    return values + [0.0] * (count - len(values))


def write_known_poses_model(database: Path, poses_model: Path, folder: Path) -> int:
    """A binary model of `database`'s cameras, rigs and frames, with `poses_model`'s poses.

    Writes cameras.bin, rigs.bin, frames.bin, images.bin and an empty
    points3D.bin into `folder`; returns the number of posed images. Images
    the plugin didn't place are left out (unregistered). Raises
    BackendError for rigs of more than one camera, which this doesn't need
    and doesn't write.
    """
    poses = read_images(poses_model)
    plugin_cameras = read_cameras(poses_model)
    with contextlib.closing(sqlite3.connect(database)) as db:
        cameras = {
            row[0]: row[1:]
            for row in db.execute("SELECT camera_id, model, width, height, params FROM cameras")
        }
        rigs = dict(db.execute("SELECT rig_id, ref_sensor_id FROM rigs WHERE ref_sensor_type = 0"))
        if db.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0]:
            raise BackendError("refining needs rigs of one camera each; this database has more")
        frame_of: dict[int, int] = {}
        frame_data: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for frame_id, data_id, sensor_id, sensor_type in db.execute(
            "SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data"
        ):
            frame_data[frame_id].append((sensor_type, sensor_id, data_id))
            if sensor_type == CAMERA_SENSOR:
                frame_of[data_id] = frame_id
        frame_rig = dict(db.execute("SELECT frame_id, rig_id FROM frames"))
        images = list(db.execute("SELECT image_id, name, camera_id FROM images"))

    posed = [(image_id, name, camera_id) for image_id, name, camera_id in images if name in poses]
    # Starting intrinsics per database camera: the plugin's, averaged over its photos.
    found: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for _image_id, name, camera_id in posed:
        plugin_camera = plugin_cameras[poses[name].camera_id]
        width = cameras[camera_id][1]
        scale = width / plugin_camera.width if plugin_camera.width else 1.0
        found[camera_id].append(tuple(v * scale for v in intrinsics(plugin_camera)))  # type: ignore[arg-type]

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for camera_id in sorted(cameras):
            model_id, width, height, blob = cameras[camera_id]
            name, count = CAMERA_MODELS[model_id]
            if found[camera_id]:
                fx, fy, cx, cy = (
                    sum(v[i] for v in found[camera_id]) / len(found[camera_id]) for i in range(4)
                )
                params = params_for(name, fx, fy, cx, cy)
            else:
                params = list(struct.unpack(f"<{count}d", blob))
            f.write(struct.pack(f"<IiQQ{count}d", camera_id, model_id, width, height, *params))
    with (folder / "rigs.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(rigs)))
        for rig_id in sorted(rigs):
            f.write(struct.pack("<IIiI", rig_id, 1, CAMERA_SENSOR, rigs[rig_id]))
    with (folder / "frames.bin").open("wb") as f, (folder / "images.bin").open("wb") as g:
        f.write(struct.pack("<Q", len(posed)))
        g.write(struct.pack("<Q", len(posed)))
        for image_id, name, camera_id in sorted(posed):
            pose = poses[name]
            frame_id = frame_of.get(image_id)
            if frame_id is None or len(frame_data[frame_id]) != 1:
                raise BackendError(f"{name}: not a frame of its own in the database")
            # One camera per rig, the reference: the rig's pose is the camera's.
            f.write(struct.pack("<II7d", frame_id, frame_rig[frame_id], *pose.qvec, *pose.tvec))
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<iIQ", CAMERA_SENSOR, camera_id, image_id))
            g.write(struct.pack("<I7dI", image_id, *pose.qvec, *pose.tvec, camera_id))
            g.write(name.encode() + b"\0")
            g.write(struct.pack("<Q", 0))
    (folder / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    return len(posed)


# --- the stages ----------------------------------------------------------------------


def match_pairs(
    sfm: colmap.Colmap,
    project: Project,
    features: StageManifest,
    poses: StageManifest,
    *,
    stage: str = "matching",
    options: colmap.MatchOptions | None = None,
    camera_groups: Mapping[str, str] | None = None,
) -> StageSpec:
    """Feature matching of the pairs the plugin's poses suggest (`pose_pairs`)."""
    pairs = pose_pairs(_model(project, poses))
    if not pairs:
        raise BackendError("the plugin's camera placement gives no photos to match")
    spec = colmap.match_listed(
        sfm,
        project,
        features,
        pairs,
        stage=stage,
        mode="pose_pairs",
        options=options,
        camera_groups=camera_groups,
    )
    return replace(spec, inputs={**spec.inputs, "poses": stage_input(poses)})


def triangulate(
    sfm: colmap.Colmap,
    project: Project,
    matching: StageManifest,
    poses: StageManifest,
    *,
    stage: str = "triangulation",
) -> StageSpec:
    """`point_triangulator`: points from the matches, the plugin's poses held."""
    stage_dir = project.stage_dir(stage)
    database = project.stage_dir(matching.stage) / colmap.DATABASE
    model = _model(project, poses)
    argv: list[str | Path] = [
        sfm.path, "point_triangulator",
        "--database_path", database,
        "--image_path", project.captures_dir,
        "--input_path", stage_dir / KNOWN_POSES,
        "--output_path", stage_dir / "sparse" / "0",
        "--clear_points", "1",
        "--refine_intrinsics", "0",
    ]  # fmt: skip

    def prepare(folder: Path) -> None:
        (folder / "sparse" / "0").mkdir(parents=True)
        write_known_poses_model(database, model, folder / KNOWN_POSES)

    return StageSpec(
        name=stage,
        backend=sfm.backend,
        argv=argv,
        parameters={},
        inputs={"matching": stage_input(matching), "poses": stage_input(poses)},
        parse_line=colmap.ColmapProgress(),
        prepare=prepare,
    )


def drop_weak(
    sfm: colmap.Colmap,
    project: Project,
    triangulation: StageManifest,
    *,
    stage: str = "pose-check",
    min_points: int = MIN_POINTS,
) -> StageSpec:
    """`image_filterer`: photos seeing fewer than `min_points` points leave the model.

    A camera the plugin placed badly gets almost no points when its pose is
    held (1 or 2 on the synthetic scene at 5° of error, against hundreds for
    the others), and a bundle adjustment can't pull it back from there. Left
    out, the mapper places it afresh against the others' points.
    """
    stage_dir = project.stage_dir(stage)
    argv: list[str | Path] = [
        sfm.path, "image_filterer",
        "--input_path", project.stage_dir(triangulation.stage) / "sparse" / "0",
        "--output_path", stage_dir / "sparse" / "0",
        "--min_num_observations", str(min_points),
    ]  # fmt: skip
    return StageSpec(
        name=stage,
        backend=sfm.backend,
        argv=argv,
        parameters={"min_points": min_points},
        inputs={"triangulation": stage_input(triangulation)},
        parse_line=colmap.ColmapProgress(),
        prepare=lambda folder: (folder / "sparse" / "0").mkdir(parents=True),
    )


def refine(
    sfm: colmap.Colmap,
    project: Project,
    checked: StageManifest,
    matching: StageManifest,
    *,
    total_images: int | None = None,
    stage: str = "mapping",
    threads: int | None = None,
) -> StageSpec:
    """COLMAP's `mapper`, continuing from the checked model: the refined placement.

    With every camera it can place already placed, it places the ones left
    out (`drop_weak`, or never placed by the plugin), then runs its iterative
    global refinement: bundle adjustment of poses, intrinsics and points,
    re-triangulation and outlier filtering, until it settles. One bundle
    adjustment on the held poses' points alone made things worse on the
    synthetic scene (2.9° median rotation error to 4.9°); this brought it to
    0.06°. The model lands in `<stage>/sparse/0`, where the camera placement
    is read from; `matching` is an input so the coverage notes use its database.
    """
    stage_dir = project.stage_dir(stage)
    argv: list[str | Path] = [
        sfm.path, "mapper",
        "--database_path", project.stage_dir(matching.stage) / colmap.DATABASE,
        "--image_path", project.captures_dir,
        "--input_path", project.stage_dir(checked.stage) / "sparse" / "0",
        "--output_path", stage_dir / "sparse" / "0",
    ]  # fmt: skip
    if threads:
        argv += ["--Mapper.num_threads", str(threads)]
    return StageSpec(
        name=stage,
        backend=sfm.backend,
        argv=argv,
        parameters={"continue_from": "pose-check"},
        inputs={"checked": stage_input(checked), "matching": stage_input(matching)},
        parse_line=colmap.ColmapProgress(total_images=total_images),
        prepare=lambda folder: (folder / "sparse" / "0").mkdir(parents=True),
    )


def _model(project: Project, poses: StageManifest) -> Path:
    model = colmap.best_model(project.stage_dir(poses.stage) / "sparse")
    if model is None:
        raise BackendError("the camera placement plugin wrote no COLMAP model")
    return model
