# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from scripts import printing_script

from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.cli import main
from ez2digitize.pipeline import Tools


def test_new_import_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    photos, masks = tmp_path / "photos", tmp_path / "masks"
    photos.mkdir()
    masks.mkdir()
    for name in ("a.jpg", "b.jpg", "notes.txt"):
        (photos / name).write_bytes(name.encode())
    (masks / "a.jpg.png").write_bytes(b"m")

    assert main(["new", str(tmp_path / "scan"), "--name", "Skull"]) == 0
    assert main(["import", str(tmp_path / "scan"), str(photos), "--masks", str(masks)]) == 0
    assert main(["status", str(tmp_path / "scan")]) == 0
    out = capsys.readouterr().out
    assert "created project 'Skull'" in out
    assert "2 photos -> capture" in out and "skipped (not a photo): notes.txt" in out
    assert "1 of 2 masks imported" in out
    assert "folder   2 images, 0 videos" in out
    assert "features    -" in out


def test_photo_checks_and_exclusion(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from PIL import Image

    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(6):
        Image.new("RGB", (1200, 1000), (i * 40, 90, 90)).save(photos / f"{i}.jpg")
    Image.new("RGB", (300, 200)).save(photos / "Preview.jpg")
    (photos / "broken.jpg").write_bytes(b"not an image")
    scan = str(tmp_path / "scan")
    assert main(["new", scan]) == 0
    assert main(["import", scan, str(photos)]) == 0
    out = capsys.readouterr().out
    assert "error [" in out and "broken.jpg can't be read" in out
    assert "Preview.jpg is a different size from the others (1200×1000)" in out

    assert main(["photos", scan, "--exclude", "Preview.jpg", "broken.jpg"]) == 0
    out = capsys.readouterr().out
    assert "left out: " in out and "/Preview.jpg" in out
    assert "broken" not in out.split("photo checks:")[1]
    assert main(["status", scan]) == 0
    assert "6 images, 0 videos (2 left out)" in capsys.readouterr().out

    assert main(["photos", scan, "--include", "Preview.jpg"]) == 0
    assert "brought back" in capsys.readouterr().out
    assert main(["photos", scan, "--exclude", "nope.jpg"]) == 1
    assert "no photo 'nope.jpg'" in capsys.readouterr().err


def test_import_video(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_ffmpeg: FFmpeg
) -> None:
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"video")
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "walkaround.mp4").write_bytes(b"video")
    scan = str(tmp_path / "scan")
    assert main(["new", scan]) == 0
    args = ["import", scan, str(clip), "--frames", "10", "--ffmpeg", str(fake_ffmpeg.path)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Extracting frames: 100%" in out
    assert "10 frames, the sharpest of 40 extracted -> capture" in out
    assert main(["import", scan, str(folder)]) == 1
    assert "import videos one by one, e.g. walkaround.mp4" in capsys.readouterr().err
    assert main(["import", scan, str(clip), "--frames", "0"]) == 2


def test_errors_are_reported_not_raised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
    (tmp_path / "x").mkdir()
    assert main(["import", str(tmp_path / "p"), "a", "b", "--masks", str(tmp_path)]) == 2


def test_export_without_mesh_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", str(tmp_path / "scan")]) == 0
    assert main(["export", str(tmp_path / "scan"), "--formats", "glb"]) == 1
    assert "build the mesh first" in capsys.readouterr().err


def test_format_list_parsing() -> None:
    import argparse

    from ez2digitize.cli import _formats

    assert _formats("obj,GLB") == ("obj", "glb")
    assert _formats("stl,3MF,points") == ("stl", "3mf", "points")
    assert _formats("none") == ()
    with pytest.raises(argparse.ArgumentTypeError, match="unknown format 'step'"):
        _formats("obj,step")


def _tool(path: Path, text: str) -> Path:
    return printing_script(path, text)


def test_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    colmap = _tool(tmp_path / "colmap", "COLMAP 4.2.1 (Commit abc)")
    for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh",
                 "TextureMesh"):  # fmt: skip
        _tool(tmp_path / "mvs" / name, "12:00:00 [App     ] OpenMVS x64 v2.4.0")
    monkeypatch.setenv("EZ2D_COLMAP", str(colmap))
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "mvs"))
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert f"COLMAP 4.2.1: ok, {colmap}" in out and "OpenMVS 2.4.0: ok" in out
    assert "Masking model isnet-general-use: not downloaded yet (179 MB" in out

    _tool(colmap, "COLMAP 3.9.1 -- SfM")
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "nothing"))
    assert main(["check"]) == 1
    out = capsys.readouterr().out
    assert "not the tested version 4.2.1" in out and "OpenMVS: OpenMVS not found" in out


def test_commands_list_matches_parser() -> None:
    from ez2digitize.cli import commands

    assert set(commands()) == {
        "new", "import", "flip", "upload", "watch", "photos", "masks", "crop", "scale",
        "markers", "orient", "run", "export", "check", "diagnostics", "plugins", "licenses",
        "status",
    }  # fmt: skip


def test_run_with_quality(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_tools: Tools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ez2digitize.backends import colmap, openmvs
    from ez2digitize.core.project import Project

    monkeypatch.setattr(colmap, "locate", lambda _path: fake_tools.colmap)
    monkeypatch.setattr(openmvs, "locate", lambda _path: fake_tools.openmvs)

    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (photos / name).write_bytes(name.encode())
    scan = str(tmp_path / "scan")
    assert main(["new", scan]) == 0
    assert main(["import", scan, str(photos)]) == 0
    tools: list[str] = []
    assert main(["run", scan, "--quality", "fast", "--export", "none", *tools]) == 0
    out = capsys.readouterr().out
    assert "quality: Fast" in out and "Dense point cloud: 1/4 size photos" in out
    assert Project.open(Path(scan)).preset == "fast"
    # The project remembers it; --level overrides one value.
    assert main(["run", scan, "--level", "0", "--export", "none", *tools]) == 0
    out = capsys.readouterr().out
    assert "quality: Fast" in out and "Dense point cloud: full size photos" in out


def test_licenses(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["licenses"]) == 0
    out = capsys.readouterr().out
    assert "Source for the bundled backends" in out and "No bundled backends" in out
    assert main(["licenses", "--gpl"]) == 0
    assert "GNU GENERAL PUBLIC LICENSE" in capsys.readouterr().out


def test_masks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_mask_worker: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    project = tmp_path / "p"
    photos = tmp_path / "photos"
    photos.mkdir()
    for n, name in enumerate(("a.jpg", "b.jpg")):
        Image.new("RGB", (40, 30), (n * 90, 0, 0)).save(photos / name)
    assert main(["new", str(project)]) == 0
    assert main(["masks", str(project)]) == 1  # no photos yet
    assert main(["import", str(project), str(photos)]) == 0
    capsys.readouterr()

    monkeypatch.setenv("FAKE_COVERAGE", "b.jpg=0")
    assert main(["masks", str(project)]) == 0
    out = capsys.readouterr().out
    assert "2 new masks, 1 dropped (nothing found)" in out
    assert "masks: 1 automatic, 1 dropped" in out
    assert "/b.jpg (dropped): nothing found" in out

    assert main(["masks", str(project), "--drop", "a.jpg"]) == 0
    assert "masks: 2 dropped" in capsys.readouterr().out
    assert main(["masks", str(project), "--restore", "a.jpg", "b.jpg"]) == 0
    assert "masks: 2 automatic" in capsys.readouterr().out
    assert main(["masks", str(project)]) == 0
    assert "masks unchanged" in capsys.readouterr().out

    imported = tmp_path / "masks"
    imported.mkdir()
    Image.new("L", (40, 30), 255).save(imported / "a.jpg.png")
    assert main(["masks", str(project), "--import", str(imported)]) == 0
    out = capsys.readouterr().out
    assert "1 of 2 masks imported" in out and "1 automatic, 1 imported" in out
    assert main(["masks", str(project), "--clear"]) == 0
    out = capsys.readouterr().out
    assert "1 automatic masks removed" in out and "masks: 1 imported, 1 none" in out
    assert main(["masks", str(project), "--status"]) == 0


def test_two_sided_import_and_flip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from PIL import Image

    from ez2digitize.core.capture import list_bundles
    from ez2digitize.core.project import Project

    project = tmp_path / "p"
    for side in ("top", "under"):
        (tmp_path / side).mkdir()
        Image.new("RGB", (40, 30), (90, 0, 0)).save(tmp_path / side / f"{side}.jpg")
    assert main(["new", str(project)]) == 0
    assert main(["import", str(project), str(tmp_path / "top")]) == 0
    assert "two-sided" not in capsys.readouterr().out
    assert main(["import", str(project), str(tmp_path / "under"), "--flipped"]) == 0
    out = capsys.readouterr().out
    assert "(turned over)" in out
    assert "two-sided scan: 1 photos of the first side, 1 turned over" in out
    assert "to do: 2 of 2 photos have no mask" in out

    bundles = list_bundles(Project.open(project))
    assert [b.flipped for b in bundles] == [False, True]
    assert main(["flip", str(project), bundles[1].id, "--undo"]) == 0
    assert "the first side" in capsys.readouterr().out
    assert not any(b.flipped for b in list_bundles(Project.open(project)))
    assert main(["flip", str(project), bundles[0].id]) == 0
    assert main(["status", str(project)]) == 0
    assert "(turned over)" in capsys.readouterr().out
    assert main(["flip", str(project), "nope"]) == 1


def test_crop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ez2digitize.core.files import write_json_atomic
    from ez2digitize.core.project import Project
    from ez2digitize.core.stage import MANIFEST_FILE, Backend, StageManifest

    project = tmp_path / "p"
    assert main(["new", str(project)]) == 0
    assert main(["crop", str(project)]) == 0
    assert "no crop box" in capsys.readouterr().out
    assert main(["crop", str(project), "--set", "0", "0", "0", "1", "1", "1"]) == 1
    assert "place the cameras first" in capsys.readouterr().err

    mapping = Project.open(project).stage_dir("mapping")
    mapping.mkdir(parents=True)
    manifest = StageManifest(
        stage="mapping", run_id="m1", status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(mapping / MANIFEST_FILE, manifest.to_dict())
    args = ["crop", str(project), "--set", "1", "2", "3", "0.5", "0.5", "1", "--yaw", "15"]
    assert main(args) == 0
    assert "crop box: centre (1, 2, 3), size (1, 1, 2), turned 15.0°" in capsys.readouterr().out
    assert main(["crop", str(project), "--auto"]) == 1  # no sparse points in this project
    assert main(["crop", str(project), "--clear"]) == 0
    assert main(["crop", str(project)]) == 0
    assert "no crop box" in capsys.readouterr().out


def test_scale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ez2digitize.core.files import write_json_atomic
    from ez2digitize.core.project import Project
    from ez2digitize.core.stage import MANIFEST_FILE, Backend, StageManifest

    project = tmp_path / "p"
    assert main(["new", str(project)]) == 0
    assert main(["scale", str(project)]) == 0
    assert "no scale" in capsys.readouterr().out
    points = ["--points", "0", "0", "0", "0", "2", "0"]
    assert main(["scale", str(project), *points, "--distance", "50"]) == 1
    assert "place the cameras first" in capsys.readouterr().err

    mapping = Project.open(project).stage_dir("mapping")
    mapping.mkdir(parents=True)
    manifest = StageManifest(
        stage="mapping", run_id="m1", status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(mapping / MANIFEST_FILE, manifest.to_dict())
    assert main(["scale", str(project), *points]) == 1
    assert "--distance" in capsys.readouterr().err
    assert main(["scale", str(project), *points, "--distance", "50"]) == 0
    out = capsys.readouterr().out
    assert "scale: 50 mm between the two points (25 mm per unit)" in out
    assert "points (0, 0, 0) and (0, 2, 0)" in out
    # The real distance again, measured more carefully: same points.
    assert main(["scale", str(project), "--distance", "48"]) == 0
    assert "(24 mm per unit)" in capsys.readouterr().out
    same = ["--points", "1", "1", "1", "1", "1", "1", "--distance", "5"]
    assert main(["scale", str(project), *same]) == 1
    assert "two different points" in capsys.readouterr().err
    assert main(["scale", str(project), "--clear"]) == 0
    assert main(["scale", str(project)]) == 0
    assert "no scale" in capsys.readouterr().out


def test_orient(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ez2digitize.core.files import write_json_atomic
    from ez2digitize.core.project import Project
    from ez2digitize.core.stage import MANIFEST_FILE, Backend, StageManifest

    project = tmp_path / "p"
    assert main(["new", str(project)]) == 0
    assert main(["orient", str(project), "--tilt", "x"]) == 1
    assert "place the cameras first" in capsys.readouterr().err

    mapping = Project.open(project).stage_dir("mapping")
    mapping.mkdir(parents=True)
    manifest = StageManifest(
        stage="mapping", run_id="m1", status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(mapping / MANIFEST_FILE, manifest.to_dict())
    assert main(["orient", str(project)]) == 0
    assert "the photos don't say which way is up" in capsys.readouterr().out
    assert main(["orient", str(project), "--tilt", "x", "--turn", "30"]) == 0
    out = capsys.readouterr().out
    assert "corrected, turned 30°; up is (0.000, 0.000, -1.000)" in out
    level = ["--level", "0", "0", "0", "1", "0", "0", "0", "0", "1"]
    assert main(["orient", str(project), *level]) == 1  # no cameras to tell the side
    assert "has no cameras" in capsys.readouterr().err
    assert main(["orient", str(project), "--auto"]) == 0
    assert main(["orient", str(project)]) == 0
    assert "the photos don't say" in capsys.readouterr().out


def test_watch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    synced = tmp_path / "Camera"
    synced.mkdir()
    for name in ("IMG_1.jpg", "IMG_2.jpg"):
        (synced / name).write_bytes(name.encode())  # synced a moment ago
    assert main(["new", str(tmp_path / "scan")]) == 0
    # The photos already there from the last 10 minutes; settled at once.
    args = ["watch", str(tmp_path / "scan"), str(synced), "--since", "10", "--settle", "0"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "2 new photo(s) ready." in out and "2 files -> capture" in out
    # Again: nothing new, so nothing to import (the same photos are left out).
    assert main(args) == 1
    assert "all 2 photo(s) are already in the project" in capsys.readouterr().err
    assert main(["watch", str(tmp_path / "scan"), str(tmp_path / "nowhere")]) == 1
