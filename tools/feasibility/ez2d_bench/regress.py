# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression check: today's benchmark runs against a saved reference.

Reconstruction is not deterministic, so nothing is compared byte for byte.
`save` keeps the metrics of every finished run of a results folder, and
each OpenMVS run's mesh with its cameras, in a reference folder. `check`
runs the same comparison on new results: each run must still finish, and
its numbers must stay within the reference's tolerances:

- camera placement: registered photos, mean reprojection error, points,
  and no split into more models;
- mesh: face count, and the mesh itself against the reference mesh after
  aligning the two by their cameras (Umeyama, then ICP, as `eval` does):
  the Chamfer distance and the bounding box, as a share of the reference
  mesh's diagonal;
- splats: PSNR on the evaluation views.

Run it on the same machine as the reference: a slower run is only a
warning, but a different machine makes the times meaningless.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ez2d_bench import evaluate
from ez2d_bench.evaluate import Camera, EvalError, Mesh
from ez2d_bench.record import RECORD_NAME, RunRecord
from ez2digitize.backends.colmap_model import read_images

REFERENCE_NAME = "reference.json"
FORMAT = 1


class RegressError(Exception):
    """The results or the reference can't be compared; the message says why."""


@dataclass
class Tolerances:
    """How far a run may move from the reference; saved with it, editable there."""

    registered_drop_pct: float = 5.0  # of the photos; always at least one photo
    reprojection_factor: float = 1.2  # mean error may grow to ref * factor + slack
    reprojection_slack_px: float = 0.1
    points_drop_pct: float = 25.0
    faces_change_pct: float = 25.0  # either way
    psnr_drop_db: float = 0.5
    chamfer_pct: float = 1.0  # of the reference mesh's bounding-box diagonal
    box_pct: float = 5.0  # each side of the box, same unit
    slower_factor: float = 1.5  # only a warning
    samples: int = 100_000  # points sampled per mesh

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tolerances:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise RegressError(f"unknown tolerances in the reference: {sorted(unknown)}")
        return cls(**data)


@dataclass
class Finding:
    run: str  # dataset/kind/label
    check: str
    level: str  # ok | warn | fail
    message: str


@dataclass
class Result:
    findings: list[Finding]
    new_runs: list[str]

    @property
    def failed(self) -> bool:
        return any(f.level == "fail" for f in self.findings)

    def markdown(self) -> str:
        fails = sum(f.level == "fail" for f in self.findings)
        warns = sum(f.level == "warn" for f in self.findings)
        verdict = "FAILED" if fails else "passed"
        lines = [f"# Regression check: {verdict}", ""]
        lines.append(f"{fails} failed, {warns} warnings, {len(self.findings)} checks.")
        lines += ["", "| Run | Check | Result | Detail |", "|---|---|---|---|"]
        marks = {"ok": "ok", "warn": "warning", "fail": "**FAILED**"}
        for f in self.findings:
            detail = f.message.replace("|", "\\|")
            lines.append(f"| {f.run} | {f.check} | {marks[f.level]} | {detail} |")
        if self.new_runs:
            lines += ["", "Not in the reference (not checked): " + ", ".join(self.new_runs)]
        return "\n".join(lines) + "\n"


# --- finding runs ------------------------------------------------------------------


def find_runs(results: Path, machine: str | None = None) -> dict[str, tuple[Path, RunRecord]]:
    """Every run under `results`, by dataset/kind/label."""
    runs: dict[str, tuple[Path, RunRecord]] = {}
    for path in sorted(results.rglob(RECORD_NAME)):
        record = RunRecord.from_dict(json.loads(path.read_text()))
        if machine is not None and record.machine != machine:
            continue
        key = f"{record.dataset}/{record.kind}/{record.label}"
        if key in runs:
            raise RegressError(
                f"{key} was run on more than one machine ({runs[key][1].machine}, "
                f"{record.machine}); choose one with --machine"
            )
        runs[key] = (path.parent, record)
    return runs


def run_cameras(run_dir: Path, record: RunRecord) -> dict[str, Camera]:
    """The cameras an OpenMVS run's mesh was built from: its COLMAP run's model."""
    source = record.config.get("source")
    model = run_dir.parents[1] / "colmap" / str(source) / "dense" / "sparse"
    if not (model / "images.bin").is_file():
        raise RegressError(f"{record.label}: no camera model at {model}")
    cameras = {}
    for name, pose in read_images(model).items():
        rotation = evaluate._quaternion_to_matrix(np.asarray(pose.qvec, float))
        cameras[Path(name).name] = Camera(-rotation.T @ np.asarray(pose.tvec, float), rotation)
    return cameras


def _mesh_path(record: RunRecord) -> Path | None:
    result = record.metrics.get("result")
    return Path(result) if record.kind == "openmvs" and result else None


# --- saving ------------------------------------------------------------------------


def write_ply(mesh: Mesh, path: Path) -> None:
    """Geometry only, binary: the reference keeps no textures."""
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(mesh.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = np.zeros(len(mesh.faces), np.dtype([("n", "u1"), ("v", "<i4", (3,))]))
    faces["n"] = 3
    faces["v"] = mesh.faces
    path.write_bytes(header.encode() + mesh.vertices.astype("<f4").tobytes() + faces.tobytes())


def _write_cameras(cameras: dict[str, Camera], path: Path) -> None:
    data = {
        "units": "reference",
        "cameras": {
            name: {"center": c.center.tolist(), "rotation": c.rotation.tolist()}
            for name, c in sorted(cameras.items())
        },
    }
    path.write_text(json.dumps(data) + "\n")


def save(
    results: Path, reference: Path, *, machine: str | None = None, force: bool = False
) -> list[str]:
    """Keep the finished runs of `results` as the reference. Returns notes."""
    target = reference / REFERENCE_NAME
    if target.exists() and not force:
        raise RegressError(f"{target} exists; pass --force to replace it")
    runs = find_runs(results, machine)
    if not runs:
        raise RegressError(f"no runs in {results}")
    meshes = reference / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    notes = []
    saved: dict[str, Any] = {}
    machines, versions = set(), {}
    for key, (run_dir, record) in runs.items():
        if not record.ok:
            notes.append(f"{key}: {record.status}, left out of the reference")
            continue
        entry: dict[str, Any] = {
            "kind": record.kind,
            "metrics": {k: v for k, v in record.metrics.items() if k != "result"},
            "wall_s": record.total_wall_s(),
            "tool_versions": record.tool_versions,
        }
        mesh_path = _mesh_path(record)
        if mesh_path is not None:
            stem = key.replace("/", "--")
            try:
                write_ply(evaluate.read_mesh(mesh_path), meshes / f"{stem}.ply")
                _write_cameras(run_cameras(run_dir, record), meshes / f"{stem}.cameras.json")
            except (EvalError, OSError, RegressError) as exc:
                raise RegressError(f"{key}: can't keep its mesh: {exc}") from exc
            entry["mesh"] = f"meshes/{stem}.ply"
            entry["cameras"] = f"meshes/{stem}.cameras.json"
        saved[key] = entry
        machines.add(record.machine)
        versions.update({k: v for k, v in record.tool_versions.items() if v})
    if not saved:
        raise RegressError(f"no finished runs in {results}")
    data = {
        "format": FORMAT,
        "saved": datetime.now(UTC).isoformat(timespec="seconds"),
        "machines": sorted(machines),
        "tool_versions": versions,
        "tolerances": asdict(Tolerances()),
        "runs": saved,
    }
    target.write_text(json.dumps(data, indent=2) + "\n")
    return notes


def load_reference(reference: Path) -> tuple[Tolerances, dict[str, Any]]:
    path = reference / REFERENCE_NAME
    if not path.is_file():
        raise RegressError(f"no reference at {path}; make one with `regress save`")
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise RegressError(f"{path}: unknown format {data.get('format')!r}")
    return Tolerances.from_dict(data.get("tolerances", {})), data["runs"]


# --- checking ----------------------------------------------------------------------


def check(results: Path, reference: Path, *, machine: str | None = None) -> Result:
    tolerances, expected = load_reference(reference)
    runs = find_runs(results, machine)
    findings: list[Finding] = []
    for key, ref in expected.items():
        if key not in runs:
            findings.append(Finding(key, "finished", "fail", "not run"))
            continue
        run_dir, record = runs[key]
        if not record.ok:
            message = f"{record.status}: {record.reason}" if record.reason else record.status
            findings.append(Finding(key, "finished", "fail", message))
            continue
        findings.append(Finding(key, "finished", "ok", "finished"))
        findings += check_metrics(key, record, ref, tolerances)
        if ref.get("mesh"):
            findings.append(_check_mesh(key, run_dir, record, reference, ref, tolerances))
    new = sorted(k for k in runs if k not in expected)
    return Result(findings, new)


def check_metrics(
    key: str, record: RunRecord, ref: dict[str, Any], tol: Tolerances
) -> list[Finding]:
    """The number checks for one run (no mesh comparison)."""
    now, then = record.metrics, ref["metrics"]
    out: list[Finding] = []

    def add(check: str, passed: bool, message: str, level: str = "fail") -> None:
        out.append(Finding(key, check, "ok" if passed else level, message))

    def both(name: str) -> tuple[float, float] | None:
        if then.get(name) is None:
            return None
        if now.get(name) is None:
            add(name, False, "missing (the reference has it)")
            return None
        return float(now[name]), float(then[name])

    if pair := both("registered_images"):
        total = float(then.get("total_images") or pair[1])
        allowed = max(1, math.floor(total * tol.registered_drop_pct / 100))
        add(
            "registered photos",
            pair[0] >= pair[1] - allowed,
            f"{pair[0]:.0f} (reference {pair[1]:.0f}, at most {allowed} fewer)",
        )
    if pair := both("mean_reprojection_error_px"):
        limit = pair[1] * tol.reprojection_factor + tol.reprojection_slack_px
        add(
            "reprojection error",
            pair[0] <= limit,
            f"{pair[0]:.3f} px (reference {pair[1]:.3f}, at most {limit:.3f})",
        )
    if pair := both("points"):
        least = pair[1] * (1 - tol.points_drop_pct / 100)
        add("sparse points", pair[0] >= least, f"{pair[0]:.0f} (reference {pair[1]:.0f})")
    if pair := both("models"):
        add(
            "models",
            pair[0] <= pair[1],
            f"{pair[0]:.0f} (reference {pair[1]:.0f}; more is a split)",
        )
    for name, label in (
        ("mesh_faces", "mesh faces"),
        ("refined_faces", "refined faces"),
        ("textured_faces", "textured faces"),
    ):
        if pair := both(name):
            change = abs(pair[0] - pair[1]) / max(pair[1], 1) * 100
            add(
                label,
                change <= tol.faces_change_pct,
                f"{pair[0]:.0f} (reference {pair[1]:.0f}, {change:.0f}% off)",
            )
    if pair := both("eval_psnr"):
        add(
            "splat PSNR",
            pair[0] >= pair[1] - tol.psnr_drop_db,
            f"{pair[0]:.2f} dB (reference {pair[1]:.2f}, at most {tol.psnr_drop_db} lower)",
        )
    wall, ref_wall = record.total_wall_s(), float(ref.get("wall_s") or 0)
    if ref_wall > 0:
        add(
            "time",
            wall <= ref_wall * tol.slower_factor,
            f"{wall:.0f}s (reference {ref_wall:.0f}s)",
            level="warn",
        )
    return out


def compare_meshes(
    mesh: Mesh,
    cameras: dict[str, Camera],
    ref_mesh: Mesh,
    ref_cameras: dict[str, Camera],
    tol: Tolerances,
) -> tuple[bool, str]:
    """Chamfer distance and bounding box against the reference mesh, aligned."""
    report = evaluate.evaluate(mesh, cameras, ref_mesh, ref_cameras, samples=tol.samples)
    a = report.alignment
    aligned = a["scale"] * mesh.vertices @ np.asarray(a["rotation"]).T + np.asarray(
        a["translation"]
    )
    diagonal = report.diagonal
    box_off = np.abs(
        np.concatenate(
            [aligned.min(0) - ref_mesh.vertices.min(0), aligned.max(0) - ref_mesh.vertices.max(0)]
        )
    ).max()
    chamfer_pct = report.chamfer / diagonal * 100
    box_pct = float(box_off) / diagonal * 100
    passed = chamfer_pct <= tol.chamfer_pct and box_pct <= tol.box_pct
    message = (
        f"Chamfer {chamfer_pct:.2f}% of the diagonal (at most {tol.chamfer_pct}%), "
        f"box {box_pct:.1f}% off (at most {tol.box_pct}%)"
    )
    return passed, message


def _check_mesh(
    key: str,
    run_dir: Path,
    record: RunRecord,
    reference: Path,
    ref: dict[str, Any],
    tol: Tolerances,
) -> Finding:
    mesh_path = _mesh_path(record)
    if mesh_path is None or not mesh_path.is_file():
        return Finding(key, "mesh", "fail", "the run kept no mesh")
    try:
        _units, ref_cameras = evaluate.read_ground_truth_cameras(reference / ref["cameras"])
        passed, message = compare_meshes(
            evaluate.read_mesh(mesh_path),
            run_cameras(run_dir, record),
            evaluate.read_mesh(reference / ref["mesh"]),
            ref_cameras,
            tol,
        )
    except (EvalError, RegressError, OSError) as exc:
        return Finding(key, "mesh", "fail", f"can't compare: {exc}")
    return Finding(key, "mesh", "ok" if passed else "fail", message)
