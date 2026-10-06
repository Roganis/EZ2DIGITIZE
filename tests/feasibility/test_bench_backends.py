# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plumbing tests for the OpenMVS and Brush stages, using fake executables.

The fakes print help text with the options the real tools have and write the
output files the real tools write, so these tests run in CI without them.
"""

import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ez2d_bench import brush, openmvs, synthetic
from ez2d_bench.execute import Executor
from ez2d_bench.plan import BrushRun, ColmapRun, Dataset, OpenMVSRun, Plan
from ez2d_bench.record import Recorder, RunRecord
from ez2d_bench.tools import Toolbox

FAKE_TOOL = """\
import pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(sys.argv[0] + ".calls", "a") as fh:
    fh.write(" ".join(args) + "\\n")
HELP = {
    "DensifyPointCloud": "--resolution-level arg\\n--mask-path arg\\n--ignore-mask-label arg",
    "ReconstructMesh": "-p [ --pointcloud-file ] arg",
    "RefineMesh": "-m [ --mesh-file ] arg",
    "TextureMesh": "-m [ --mesh-file ] arg\\n--export-type arg",
    "brush_app": "--total-steps <N>\\n--export-path <PATH>",
}
if "--help" in args:
    print(HELP.get(name, ""))
    sys.exit(0)
if "--version" in args:
    print("brush 0.0-fake")
    sys.exit(0)
def opt(flag):
    return args[args.index(flag) + 1] if flag in args else None
PLY = b"ply\\nformat ascii 1.0\\nelement vertex 5\\nelement face 2\\nend_header\\n"
out = opt("-o")
if out:
    p = pathlib.Path(out)
    p.write_text("mvs")
    if name in ("DensifyPointCloud", "ReconstructMesh", "RefineMesh"):
        p.with_suffix(".ply").write_bytes(PLY)
    if name == "TextureMesh":
        p.with_suffix(".obj").write_text("v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3\\n")
        (p.parent / (p.stem + "_material_00_map_Kd.png")).write_bytes(b"png")
if name == "brush_app":
    pathlib.Path(opt("--export-path"), "export_2000.ply").write_bytes(PLY)
"""

PINHOLE_CAMERAS = "1 PINHOLE 64 48 50 50 32 24\n"
IMAGES_TXT = "1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 0 0 0 1 b.jpg\n\n"


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    names = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh",
             "TextureMesh", "brush_app"]  # fmt: skip
    for name in names:
        exe = bin_dir / name
        exe.write_text(f"#!{sys.executable}\n{FAKE_TOOL}")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _fake_colmap_run(root: Path) -> Path:
    """The files a finished COLMAP run leaves behind, as far as later stages care."""
    run = root / "colmap" / "c1"
    for model in (run / "sparse_txt", run / "dense" / "sparse"):
        model.mkdir(parents=True)
        (model / "cameras.txt").write_text(PINHOLE_CAMERAS)
        (model / "images.txt").write_text(IMAGES_TXT)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (model / name).write_bytes(b"")
    (run / "dense" / "images").mkdir()
    return run


def _calls(bin_dir: Path, name: str) -> str:
    return (bin_dir / f"{name}.calls").read_text()


def _recorder(run_dir: Path, kind: str) -> Recorder:
    run_dir.mkdir(parents=True)
    return Recorder(run_dir, RunRecord(kind=kind, label="x", dataset="d", machine="m", config={}))


def test_openmvs_pipeline_with_masks(tmp_path: Path, fake_bin: Path) -> None:
    colmap_run = _fake_colmap_run(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.fromarray(np.full((48, 64), 255, np.uint8)).save(masks / "a.jpg.png")  # b has none
    dataset = Dataset(name="d", images=tmp_path / "images", masks=masks)
    cfg = OpenMVSRun(
        label="x", source="c1", resolution_level=2, masks=True, refine=True, texture_export="obj"
    )
    rec = _recorder(tmp_path / "openmvs" / "x", "openmvs")

    metrics = openmvs.run(rec, Toolbox(openmvs_dir=fake_bin), dataset, cfg, colmap_run)

    assert [s.name for s in rec.record.steps] == [
        "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh",
    ]  # fmt: skip
    assert metrics["dense_points"] == 5
    assert metrics["textured_faces"] == 1
    assert metrics["texture_files"] == 1
    densify = _calls(fake_bin, "DensifyPointCloud").splitlines()[-1]
    assert "--resolution-level 2" in densify
    assert "--mask-path" in densify and "--ignore-mask-label 0" in densify
    assert "--pointcloud-file" in _calls(fake_bin, "ReconstructMesh")
    assert "--export-type obj" in _calls(fake_bin, "TextureMesh")
    mask_out = rec.run_dir / "masks"
    assert sorted(p.name for p in mask_out.iterdir()) == ["a.mask.png"]
    assert any("1 images without a mask" in n for n in rec.record.notes)


def test_brush_prepares_dataset_and_finds_export(tmp_path: Path, fake_bin: Path) -> None:
    colmap_run = _fake_colmap_run(tmp_path)
    rec = _recorder(tmp_path / "brush" / "x", "brush")
    cfg = BrushRun(label="x", source="c1", args=["--total-steps", "2000"])

    metrics = brush.run(rec, Toolbox(brush=fake_bin / "brush_app"), cfg, colmap_run)

    assert metrics["splats"] == 5
    assert (rec.run_dir / "dataset" / "sparse" / "0" / "cameras.bin").exists()
    assert (rec.run_dir / "dataset" / "images").is_symlink()
    call = _calls(fake_bin, "brush_app").splitlines()[-1]
    assert "--export-path" in call and call.endswith("--total-steps 2000")


def test_executor_skips_dependents_of_missing_colmap_run(tmp_path: Path, fake_bin: Path) -> None:
    plan = Plan(
        path=tmp_path / "p.toml",
        dataset=Dataset(name="d", images=tmp_path / "nope"),
        colmap=[ColmapRun(label="c1")],
        openmvs=[OpenMVSRun(label="o1", source="c1")],
    )
    tb = Toolbox(openmvs_dir=fake_bin)
    records = Executor(plan, tmp_path / "results", "m", tb, only={"o1"}).execute()
    assert [(r.label, r.status) for r in records] == [("o1", "skipped")]


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs COLMAP on PATH")
def test_colmap_on_synthetic_scene(tmp_path: Path) -> None:
    data = tmp_path / "synthetic"
    synthetic.generate(data, views_per_ring=14)
    plan = Plan(
        path=tmp_path / "p.toml",
        dataset=Dataset(name="synthetic", images=data / "images"),
        colmap=[ColmapRun(label="c1", max_image_size=640)],
    )
    tb = Toolbox.discover()
    (record,) = Executor(plan, tmp_path / "results", "m", tb).execute()
    assert record.ok, record.reason
    assert record.metrics["registered_images"] >= 25
    assert record.metrics["mean_reprojection_error_px"] < 1.0
    run_dir = tmp_path / "results" / "m" / "synthetic" / "colmap" / "c1"
    assert (run_dir / "dense" / "sparse" / "cameras.txt").exists()

    # Running again with the same config is a no-op.
    (again,) = Executor(plan, tmp_path / "results", "m", tb).execute()
    assert again.started == record.started


def test_synthetic_photos_carry_their_focal_length(tmp_path: Path) -> None:
    """As a camera's photos do: COLMAP's prior is the 35 mm focal / 35 * the longer side."""
    synthetic.generate(tmp_path, views_per_ring=1, width=160, height=120)
    with Image.open(tmp_path / "images" / "view_000.jpg") as photo:
        focal_35mm = photo.getexif().get_ifd(0x8769)[0xA405]
    assert focal_35mm / 35 * 160 == pytest.approx(0.9 * 160, rel=0.02)


def test_eval_psnr_against_held_out_photos(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    photo = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(photo).save(images / "view_008.png")
    for step, noise in ((100, 40), (300, 4)):
        eval_dir = tmp_path / "run" / f"eval_{step}"
        eval_dir.mkdir(parents=True)
        noisy = np.clip(photo.astype(int) + rng.integers(-noise, noise + 1, photo.shape), 0, 255)
        Image.fromarray(noisy.astype(np.uint8)).save(eval_dir / "view_008.png")
    result = brush.eval_psnr(tmp_path / "run", images)
    assert result["eval_step"] == 300  # latest eval only
    assert result["eval_views"] == 1
    assert 35 < result["eval_psnr"] < 50
    assert brush.psnr(photo, photo) == float("inf")
