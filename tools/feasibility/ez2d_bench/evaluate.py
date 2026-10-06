# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure a reconstruction against ground truth, the way MVS benchmarks do.

Inputs: an EZ2DIGITIZE project (its mesh and the COLMAP cameras it was made
from), a ground-truth mesh, and the ground-truth cameras (centre and
world-to-camera rotation per image name).

1. Alignment. A reconstruction has arbitrary scale, rotation and position.
   A similarity transform is fitted between the reconstructed and the true
   camera centres (Umeyama), then refined by rigid ICP between the two
   surfaces (trimmed: the worst matches are ignored).
2. Sampling. Points are sampled uniformly by area on both meshes.
3. Metrics (DTU / Tanks and Temples style), in ground-truth units:
   - accuracy: distance from reconstruction samples to the ground truth;
   - completeness: distance from ground-truth samples to the reconstruction;
   - Chamfer: the mean of the two means;
   - precision / recall / F-score at thresholds (by default 0.5, 1 and 2%
     of the ground-truth bounding-box diagonal).
   Reconstruction samples far outside the ground truth's bounding box
   (background, turntable) are left out of accuracy, and counted.
4. Poses: per camera, centre error and rotation error after alignment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ez2digitize.backends.colmap_model import read_images
from ez2digitize.core.stage import load_manifest

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip
MESH_OUTPUTS = (
    ("texture", "scene_textured.ply"),
    ("refine", "scene_refined.ply"),
    ("mesh", "scene_mesh.ply"),
)


ICP_POINTS = 20_000


class EvalError(Exception):
    pass


@dataclass
class Mesh:
    vertices: F64  # (n, 3)
    faces: I64  # (m, 3)


@dataclass
class Camera:
    center: F64  # (3,)
    rotation: F64  # (3, 3), world to camera


@dataclass
class Report:
    units: str
    diagonal: float
    accuracy_mean: float
    accuracy_median: float
    completeness_mean: float
    completeness_median: float
    chamfer: float
    thresholds: list[dict[str, float]]
    outside_share: float
    alignment: dict[str, Any]
    poses: dict[str, float]
    samples: int
    mesh: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def markdown(self) -> str:
        u = self.units
        lines = [
            f"Mesh: `{self.mesh}`  ",
            f"Ground-truth diagonal: {self.diagonal:.4g} {u}; {self.samples} samples per surface",
            "",
            "| Metric | Mean | Median |",
            "|---|---|---|",
            f"| Accuracy | {self.accuracy_mean:.4g} {u} | {self.accuracy_median:.4g} {u} |",
            f"| Completeness | {self.completeness_mean:.4g} {u} "
            f"| {self.completeness_median:.4g} {u} |",
            f"| Chamfer | {self.chamfer:.4g} {u} | |",
            "",
            "| Threshold | Precision | Recall | F-score |",
            "|---|---|---|---|",
        ]
        for t in self.thresholds:
            lines.append(
                f"| {t['tau']:.4g} {u} ({t['tau_pct']:.2g}% of diag.) | {t['precision']:.1%} "
                f"| {t['recall']:.1%} | {t['f_score']:.1%} |"
            )
        p = self.poses
        lines += [
            "",
            f"Cameras: {p['matched']:.0f} matched; centre error median {p['center_median']:.4g} "
            f"{u} (max {p['center_max']:.4g}); rotation error median "
            f"{p['rotation_median_deg']:.3g}° (max {p['rotation_max_deg']:.3g}°)",
            f"Reconstruction outside the ground truth's box (not in accuracy): "
            f"{self.outside_share:.1%}",
            f"Alignment: scale {self.alignment['scale']:.5g}, "
            f"ICP {self.alignment['icp_iterations']} iterations, "
            f"final RMS {self.alignment['icp_rms']:.4g} {u}",
        ]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


# --- reading -------------------------------------------------------------------


def read_mesh(path: Path) -> Mesh:
    """Triangle mesh from PLY (binary little-endian or ASCII) or OBJ."""
    if path.suffix.lower() == ".obj":
        return _read_obj(path)
    if path.suffix.lower() == ".ply":
        return _read_ply(path)
    raise EvalError(f"{path}: only .ply and .obj meshes are supported")


def _read_obj(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return Mesh(np.asarray(vertices, float), np.asarray(faces, np.int64).reshape(-1, 3))


def _read_ply(path: Path) -> Mesh:
    data = path.read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise EvalError(f"{path}: not a PLY file")
    header = data[:end].decode("ascii", errors="replace").splitlines()
    body_start = data.index(b"\n", end) + 1
    fmt = next(line.split()[1] for line in header if line.startswith("format"))
    elements: list[tuple[str, int, list[list[str]]]] = []
    for line in header:
        words = line.split()
        if words and words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words and words[0] == "property":
            elements[-1][2].append(words[1:])
    if fmt == "ascii":
        return _read_ply_ascii(data[body_start:].decode(errors="replace"), elements)
    if fmt != "binary_little_endian":
        raise EvalError(f"{path}: unsupported PLY format {fmt}")
    offset = body_start
    vertices: F64 = np.zeros((0, 3))
    faces: I64 = np.zeros((0, 3), np.int64)
    for name, count, props in elements:
        if all(p[0] != "list" for p in props):
            dtype = np.dtype([(p[1], "<" + PLY_TYPES[p[0]]) for p in props])
            rows = np.frombuffer(data, dtype, count, offset)
            offset += dtype.itemsize * count
            if name == "vertex":
                vertices = np.stack([rows["x"], rows["y"], rows["z"]], 1).astype(float)
            continue
        # Lists: assume triangles (as OpenMVS writes); check every count.
        fields = []
        for p in props:
            if p[0] == "list":
                n = 3 if p[3] in ("vertex_indices", "vertex_index") else 6
                fields += [(p[3] + "#n", "<" + PLY_TYPES[p[1]]),
                           (p[3], "<" + PLY_TYPES[p[2]], (n,))]  # fmt: skip
            else:
                fields.append((p[1], "<" + PLY_TYPES[p[0]]))
        dtype = np.dtype(fields)
        rows = np.frombuffer(data, dtype, count, offset)
        offset += dtype.itemsize * count
        if name == "face":
            names = rows.dtype.names or ()
            key = "vertex_indices" if "vertex_indices" in names else "vertex_index"
            if np.any(rows[key + "#n"] != 3):
                raise EvalError(f"{path}: only triangle meshes are supported")
            faces = rows[key].astype(np.int64)
    return Mesh(vertices, faces)


def _read_ply_ascii(text: str, elements: list[tuple[str, int, list[list[str]]]]) -> Mesh:
    tokens = iter(text.split())
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for name, count, props in elements:
        for _ in range(count):
            values: dict[str, Any] = {}
            for p in props:
                if p[0] == "list":
                    n = int(next(tokens))
                    values[p[3]] = [next(tokens) for _ in range(n)]
                else:
                    values[p[1]] = next(tokens)
            if name == "vertex":
                vertices.append([float(values[k]) for k in ("x", "y", "z")])
            elif name == "face":
                idx = [int(i) for i in values.get("vertex_indices", values.get("vertex_index", []))]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return Mesh(np.asarray(vertices, float), np.asarray(faces, np.int64).reshape(-1, 3))


def read_ground_truth_cameras(path: Path) -> tuple[str, dict[str, Camera]]:
    """`{"units": ..., "cameras": {name: {"center": [..], "rotation": [[..]]}}}`."""
    info = json.loads(path.read_text())
    cameras = {
        Path(name).name: Camera(np.asarray(c["center"], float), np.asarray(c["rotation"], float))
        for name, c in info["cameras"].items()
    }
    return str(info.get("units", "units")), cameras


def project_reconstruction(
    project: Path, mesh: Path | None = None
) -> tuple[Path, dict[str, Camera]]:
    """The project's final mesh and the cameras of the model it was built from."""
    stages = project / "stages"
    if mesh is None:
        for stage, name in MESH_OUTPUTS:
            candidate = stages / stage / name
            manifest = load_manifest(stages / stage)
            if candidate.is_file() and manifest is not None and manifest.succeeded:
                mesh = candidate
                break
        else:
            raise EvalError(f"{project}: no finished mesh (texture, refine or mesh step)")
    undistort = load_manifest(stages / "undistort")
    if undistort is None or not undistort.succeeded:
        raise EvalError(f"{project}: no finished undistort step to take the cameras from")
    model = stages / "mapping" / "sparse" / str(undistort.parameters.get("model"))
    cameras = {}
    for name, pose in read_images(model).items():
        rotation = _quaternion_to_matrix(np.asarray(pose.qvec, float))
        center = -rotation.T @ np.asarray(pose.tvec, float)
        cameras[Path(name).name] = Camera(center, rotation)
    return mesh, cameras


def _quaternion_to_matrix(q: F64) -> F64:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


# --- geometry ------------------------------------------------------------------


def sample_surface(mesh: Mesh, count: int, rng: np.random.Generator) -> F64:
    """`count` points uniformly distributed over the mesh's area."""
    tri = mesh.vertices[mesh.faces]  # (m, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if areas.sum() <= 0:
        raise EvalError("mesh has no area")
    chosen = rng.choice(len(areas), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[chosen]
    points: F64 = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return points


def umeyama(source: F64, target: F64, *, with_scale: bool = True) -> tuple[float, F64, F64]:
    """Least-squares similarity (s, R, t) with target ~ s R source + t."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    cov = xt.T @ xs / len(source)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / xs.var(0).sum()) if with_scale else 1.0
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def icp(
    source: F64, target_tree: cKDTree, target: F64, *, iterations: int = 50, trim: float = 0.8
) -> tuple[F64, F64, int, float]:
    """Rigid ICP: (R, t) moving `source` onto the target surface.

    Each round keeps the `trim` share of closest pairs, so background and
    holes don't pull the alignment.
    """
    rotation, translation = np.eye(3), np.zeros(3)
    moved = source
    previous = math.inf
    rms = math.inf
    done = 0
    for done in range(1, iterations + 1):  # noqa: B007 - the count is returned
        dist, idx = target_tree.query(moved, workers=-1)
        keep = dist <= np.quantile(dist, trim)
        _, r, t = umeyama(moved[keep], target[idx[keep]], with_scale=False)
        moved = moved @ r.T + t
        rotation, translation = r @ rotation, r @ translation + t
        rms = float(np.sqrt(np.mean(dist[keep] ** 2)))
        if previous - rms < 1e-7 * max(rms, 1e-12):
            break
        previous = rms
    return rotation, translation, done, rms


# --- evaluation ----------------------------------------------------------------


def evaluate(
    reconstruction: Mesh,
    recon_cameras: dict[str, Camera],
    ground_truth: Mesh,
    gt_cameras: dict[str, Camera],
    *,
    units: str = "units",
    samples: int = 200_000,
    thresholds_pct: tuple[float, ...] = (0.5, 1.0, 2.0),
    box_margin_pct: float = 10.0,
    mesh_name: str = "",
    seed: int = 0,
) -> Report:
    rng = np.random.default_rng(seed)
    names = sorted(set(recon_cameras) & set(gt_cameras))
    if len(names) < 3:
        raise EvalError(f"only {len(names)} cameras match the ground truth by name; need 3")
    src = np.stack([recon_cameras[n].center for n in names])
    dst = np.stack([gt_cameras[n].center for n in names])
    scale, rotation, translation = umeyama(src, dst)

    gt_points = sample_surface(ground_truth, samples, rng)
    rec_points = sample_surface(reconstruction, samples, rng) * scale @ rotation.T + translation
    lo, hi = ground_truth.vertices.min(0), ground_truth.vertices.max(0)
    diagonal = float(np.linalg.norm(hi - lo))
    margin = diagonal * box_margin_pct / 100
    inside = np.all((rec_points >= lo - margin) & (rec_points <= hi + margin), axis=1)
    notes = []
    if inside.sum() < 100:
        raise EvalError("almost nothing of the reconstruction lies near the ground truth")

    gt_tree = cKDTree(gt_points)
    # ICP needs a well-spread subset, not every sample.
    subset = rec_points[inside]
    if len(subset) > ICP_POINTS:
        subset = subset[rng.choice(len(subset), ICP_POINTS, replace=False)]
    r_icp, t_icp, iterations, rms = icp(subset, gt_tree, gt_points)
    rec_points = rec_points @ r_icp.T + t_icp
    inside = np.all((rec_points >= lo - margin) & (rec_points <= hi + margin), axis=1)
    rec_inside = rec_points[inside]

    accuracy = gt_tree.query(rec_inside, workers=-1)[0]
    completeness = cKDTree(rec_inside).query(gt_points, workers=-1)[0]
    thresholds = []
    for pct in thresholds_pct:
        tau = diagonal * pct / 100
        precision = float(np.mean(accuracy <= tau))
        recall = float(np.mean(completeness <= tau))
        f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        thresholds.append(
            {
                "tau": tau,
                "tau_pct": pct,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }  # fmt: skip
        )

    # Camera poses in the ground-truth frame after both alignment steps.
    total_r = r_icp @ rotation
    total_t = r_icp @ translation + t_icp
    center_errors, rotation_errors = [], []
    for n in names:
        c = scale * total_r @ recon_cameras[n].center + total_t
        center_errors.append(float(np.linalg.norm(c - gt_cameras[n].center)))
        r_cam = recon_cameras[n].rotation @ total_r.T  # world(GT) -> camera
        delta = r_cam @ gt_cameras[n].rotation.T
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(delta) - 1) / 2))))
        rotation_errors.append(angle)
    missing = len(gt_cameras) - len(names)
    if missing:
        notes.append(f"{missing} ground-truth cameras were not reconstructed")

    return Report(
        units=units,
        diagonal=diagonal,
        accuracy_mean=float(accuracy.mean()),
        accuracy_median=float(np.median(accuracy)),
        completeness_mean=float(completeness.mean()),
        completeness_median=float(np.median(completeness)),
        chamfer=float((accuracy.mean() + completeness.mean()) / 2),
        thresholds=thresholds,
        outside_share=float(1 - inside.mean()),
        alignment={
            "scale": scale,
            "rotation": total_r.tolist(),
            "translation": total_t.tolist(),
            "icp_iterations": iterations,
            "icp_rms": rms,
        },
        poses={
            "matched": float(len(names)),
            "center_median": float(np.median(center_errors)),
            "center_max": float(np.max(center_errors)),
            "rotation_median_deg": float(np.median(rotation_errors)),
            "rotation_max_deg": float(np.max(rotation_errors)),
        },
        samples=samples,
        mesh=mesh_name,
        notes=notes,
    )


def evaluate_project(
    project: Path,
    gt_mesh: Path,
    gt_cameras: Path,
    *,
    mesh: Path | None = None,
    samples: int = 200_000,
) -> Report:
    mesh_path, recon_cameras = project_reconstruction(project, mesh)
    units, truth_cameras = read_ground_truth_cameras(gt_cameras)
    return evaluate(
        read_mesh(mesh_path),
        recon_cameras,
        read_mesh(gt_mesh),
        truth_cameras,
        units=units,
        samples=samples,
        mesh_name=str(mesh_path),
    )
