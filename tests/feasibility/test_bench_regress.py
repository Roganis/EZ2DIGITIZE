# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The regression check: a results tree saved as a reference, then checked."""

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy", reason="needs --group feasibility")

import bench  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ez2d_bench import regress, synthetic  # noqa: E402
from ez2d_bench.evaluate import Mesh, read_mesh  # noqa: E402
from ez2d_bench.record import RunRecord  # noqa: E402
from ez2d_bench.runner import StepResult  # noqa: E402

DATASET = "synth"


def _cameras() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Centre and world-to-camera rotation per photo, around the synthetic box."""
    cameras = {}
    for i in range(8):
        angle = 2 * math.pi * i / 8
        eye = np.array([2 * math.cos(angle), 2 * math.sin(angle), 1.0 + 0.2 * (i % 2)])
        right, up, forward = synthetic._look_at(eye, np.array([0.0, 0.0, 0.3]))
        cameras[f"view_{i:03d}.jpg"] = (eye, np.stack([right, -up, forward]))
    return cameras


def _write_images_bin(model: Path, cameras: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    model.mkdir(parents=True, exist_ok=True)
    data = struct.pack("<Q", len(cameras))
    for n, (name, (centre, rotation)) in enumerate(sorted(cameras.items()), 1):
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        t = -rotation @ centre
        data += struct.pack("<I7dI", n, w, x, y, z, *t, 1) + name.encode() + b"\0"
        data += struct.pack("<Q", 0)
    (model / "images.bin").write_bytes(data)


def _record(kind: str, label: str, metrics: dict[str, Any], **kw: Any) -> RunRecord:
    step = StepResult("step", ["tool"], 0, kw.pop("wall_s", 100.0), 1.0, 10.0)
    return RunRecord(
        kind=kind,
        label=label,
        dataset=DATASET,
        machine=kw.pop("machine", "gre"),
        config=kw.pop("config", {}),
        status=kw.pop("status", "ok"),
        steps=[step],
        metrics=metrics,
        **kw,
    )


def _results(
    root: Path,
    *,
    move: tuple[float, np.ndarray, np.ndarray] = (1.0, np.eye(3), np.zeros(3)),
    stretch: float = 1.0,
    colmap: dict[str, Any] | None = None,
    psnr: float = 28.0,
    wall_s: float = 100.0,
) -> Path:
    """A finished COLMAP, OpenMVS and Brush run of the synthetic box.

    `move` puts the reconstruction in another frame (x' = s R x + t), as a new
    run would; `stretch` makes the mesh taller than it should be.
    """
    base = root / "gre" / DATASET
    scale, rotation, t = move
    cameras = {n: (scale * rotation @ c + t, r @ rotation.T) for n, (c, r) in _cameras().items()}
    _write_images_bin(base / "colmap" / "base" / "dense" / "sparse", cameras)
    metrics = {
        "total_images": 8, "registered_images": 8, "models": 1, "points": 4000,
        "mean_reprojection_error_px": 0.6,
    }  # fmt: skip
    metrics.update(colmap or {})
    _record("colmap", "base", metrics, wall_s=wall_s).save(base / "colmap" / "base")

    synthetic._write_ground_truth(root, {})
    box = read_mesh(root / "ground_truth.ply")
    vertices = box.vertices * [1.0, 1.0, stretch]
    mesh = root / "mesh.ply"
    regress.write_ply(Mesh(scale * vertices @ rotation.T + t, box.faces), mesh)
    openmvs = {"mesh_faces": 10, "textured_faces": 10, "result": str(mesh)}
    _record("openmvs", "base", openmvs, config={"source": "base"}, wall_s=wall_s).save(
        base / "openmvs" / "base"
    )
    brush = {"splats": 5000, "eval_psnr": psnr}
    _record("brush", "base", brush, config={"source": "base"}, wall_s=wall_s).save(
        base / "brush" / "base"
    )
    return root


def _levels(result: regress.Result) -> dict[tuple[str, str], str]:
    return {(f.run, f.check): f.level for f in result.findings}


def test_same_results_pass(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")
    reference = tmp_path / "reference"
    assert regress.save(results, reference) == []
    data = json.loads((reference / regress.REFERENCE_NAME).read_text())
    assert set(data["runs"]) == {"synth/colmap/base", "synth/openmvs/base", "synth/brush/base"}
    assert "result" not in data["runs"]["synth/openmvs/base"]["metrics"]
    assert data["runs"]["synth/openmvs/base"]["mesh"] == "meshes/synth--openmvs--base.ply"
    assert (reference / "meshes" / "synth--openmvs--base.cameras.json").is_file()

    result = regress.check(results, reference)
    assert not result.failed, result.markdown()
    assert set(_levels(result).values()) == {"ok"}
    assert ("synth/openmvs/base", "mesh") in _levels(result)
    assert "passed" in result.markdown()

    with pytest.raises(regress.RegressError, match="--force"):
        regress.save(results, reference)


def test_a_run_in_another_frame_passes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    regress.save(_results(tmp_path / "before"), reference)
    rotation = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    after = _results(tmp_path / "after", move=(3.7, rotation, np.array([5.0, -2.0, 1.0])))
    result = regress.check(after, reference)
    assert not result.failed, result.markdown()


def test_regressions_fail(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    regress.save(_results(tmp_path / "before"), reference)
    after = _results(
        tmp_path / "after",
        stretch=1.4,
        colmap={"registered_images": 6, "mean_reprojection_error_px": 0.9, "models": 2},
        psnr=27.2,
        wall_s=200.0,
    )
    result = regress.check(after, reference)
    levels = _levels(result)
    assert result.failed
    for check in ("registered photos", "reprojection error", "models"):
        assert levels["synth/colmap/base", check] == "fail"
    assert levels["synth/colmap/base", "sparse points"] == "ok"
    assert levels["synth/openmvs/base", "mesh"] == "fail"
    assert levels["synth/brush/base", "splat PSNR"] == "fail"
    assert levels["synth/brush/base", "time"] == "warn"  # slower only warns
    assert "FAILED" in result.markdown()


def test_unfinished_and_new_runs(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")
    reference = tmp_path / "reference"
    regress.save(results, reference)
    brush = results / "gre" / DATASET / "brush" / "base"
    _record("brush", "base", {}, status="failed", reason="out of memory").save(brush)
    _record("brush", "extra", {"eval_psnr": 30.0}).save(brush.parent / "extra")
    (results / "gre" / DATASET / "openmvs" / "base" / "run.json").unlink()
    result = regress.check(results, reference)
    levels = _levels(result)
    assert levels["synth/brush/base", "finished"] == "fail"
    assert levels["synth/openmvs/base", "finished"] == "fail"
    assert result.new_runs == ["synth/brush/extra"]
    assert "out of memory" in result.markdown()

    # An unfinished run is left out of a new reference.
    notes = regress.save(results, tmp_path / "second")
    assert notes == ["synth/brush/base: failed, left out of the reference"]


def test_machines_and_tolerances(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")
    other = results / "m1" / DATASET / "brush" / "base"
    _record("brush", "base", {"eval_psnr": 25.0}, machine="m1").save(other)
    with pytest.raises(regress.RegressError, match="--machine"):
        regress.find_runs(results)
    assert len(regress.find_runs(results, "m1")) == 1

    reference = tmp_path / "reference"
    regress.save(results, reference, machine="gre")
    path = reference / regress.REFERENCE_NAME
    data = json.loads(path.read_text())
    data["tolerances"]["psnr_drop_db"] = 5.0  # a looser reference
    path.write_text(json.dumps(data))
    after = _results(tmp_path / "after", psnr=24.0)
    assert not regress.check(after, reference).failed
    data["tolerances"]["typo"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(regress.RegressError, match="typo"):
        regress.check(after, reference)


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _results(tmp_path / "results")
    reference = tmp_path / "reference"
    args = ["--results", str(results)]
    assert bench.main(["regress", "check", str(reference), *args]) == 2
    assert bench.main(["regress", "save", str(reference), *args]) == 0
    out = tmp_path / "report.md"
    assert bench.main(["regress", "check", str(reference), *args, "--out", str(out)]) == 0
    assert out.read_text().startswith("# Regression check: passed")
    worse = _results(tmp_path / "worse", psnr=20.0)
    assert bench.main(["regress", "check", str(reference), "--results", str(worse)]) == 1
    assert "no reference" in capsys.readouterr().err
