# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

from ez2digitize.backends import brush
from ez2digitize.backends.brush import Brush, BrushProgress, SplatOptions
from ez2digitize.backends.common import BackendMissing
from ez2digitize.core.project import Project
from ez2digitize.core.runner import Progress
from ez2digitize.core.stage import Backend, StageManifest


def _manifest(stage: str) -> StageManifest:
    return StageManifest(
        stage=stage, run_id="u1", status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=0, cpu_s=0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip


@pytest.mark.parametrize(
    ("text", "version"),
    [("brush-cli 0.3.0\n", "0.3.0"), ("brush_app v0.4.1", "0.4.1"), ("nope", None)],
)
def test_parse_version(text: str, version: str | None) -> None:
    assert brush.parse_version(text) == version


def test_locate(fake_brush: Brush, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert brush.locate(fake_brush.path) == fake_brush
    monkeypatch.setenv(brush.ENV_VAR, str(fake_brush.path))
    assert brush.locate().supported
    with pytest.raises(BackendMissing, match="Brush not found"):
        brush.locate(tmp_path / "missing")


def test_train_spec_and_dataset(fake_brush: Brush, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    undistort = project.stage_dir("undistort")
    (undistort / "images").mkdir(parents=True)
    (undistort / "sparse").mkdir()
    (undistort / "sparse" / "cameras.bin").write_bytes(b"c")
    spec = brush.train(fake_brush, project, _manifest("undistort"), options=SplatOptions(7000))
    args = [str(a) for a in spec.argv]
    assert args[args.index("--total-steps") + 1] == "7000"
    assert args[args.index("--export-name") + 1] == "splat.ply"
    assert spec.gpu and spec.use_pty and spec.inputs == {"undistorted": "run:u1"}
    folder = project.stage_dir("splat")
    folder.mkdir()
    assert spec.prepare is not None
    spec.prepare(folder)
    dataset = folder / "dataset"
    assert (dataset / "sparse" / "0" / "cameras.bin").read_bytes() == b"c"
    assert (dataset / "images").resolve() == (undistort / "images").resolve()
    assert not (dataset / "images").readlink().is_absolute()


def test_progress() -> None:
    parse = BrushProgress()
    line = "[56s] \x1b[36m◍◍◍◍\x1b[34m\x1b[0m\x1b[0m      1200/30000      Steps (1.3/s, 6h)"
    assert parse(line) == Progress("Training splats", 0.04)
    assert parse("\x1b[34mi\x1b[0m Completed loading   \x1b[4A") == Progress("Loaded the photos")
    assert parse("\x1b[34mi\x1b[0m Completed loading") is None  # redrawn: only once
    assert parse("evaluating every 1000 steps") is None


def test_appimage_bundles_the_pinned_version() -> None:
    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "appimage.yml"
    assert f"BRUSH_VERSION: v{brush.PINNED_VERSION}" in workflow.read_text()
