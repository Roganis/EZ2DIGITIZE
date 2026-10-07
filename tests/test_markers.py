# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest
from models import marker_scene
from PIL import Image

from ez2digitize import markers, scale
from ez2digitize.backends.colmap import Colmap
from ez2digitize.cli import main
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, StageManifest

BACKEND = Colmap(Path("colmap"), "4.2.1").backend


def _succeed(project: Project, stage: str, run_id: str) -> Path:
    folder = project.stage_dir(stage)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage=stage, run_id=run_id, status="succeeded", cache_key="k", backend=BACKEND,
        command=[], parameters={}, inputs={}, started="", finished="", wall_s=1.0,
        cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())
    return folder


@pytest.fixture
def placed(tmp_path: Path) -> tuple[Project, float]:
    """A project whose cameras are placed, with 30 mm markers in the photos."""
    project = Project.create(tmp_path / "p")
    _succeed(project, "mapping", "m1")
    undistort = _succeed(project, "undistort", "u1")
    truth = marker_scene(undistort / "sparse", undistort / "images")
    return project, truth


def test_markers_drawn_are_found() -> None:
    detector = markers._detector()
    for tag_id in (0, 7, 13):
        cells = markers.marker_cells(tag_id)
        assert cells.shape == (10, 10) and cells[0].min() == 255 and cells[1, 1:9].max() == 0
        image = np.pad(np.kron(cells, np.ones((30, 30), np.uint8)), 60, constant_values=255)
        (found,) = detector.detect(image)
        assert found.tag_id == tag_id and found.hamming == 0
    with pytest.raises(markers.MarkerError):
        markers.marker_cells(5000)


def test_sheet() -> None:
    svg = markers.sheet_svg(30)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm"')
    assert "should measure 30 mm" in svg and ">100 mm<" in svg
    assert all(f">{i}</text>" in svg for i in markers.SHEET_IDS)
    with pytest.raises(markers.MarkerError, match="choose 10 to 36 mm"):
        markers.sheet_svg(80)


def test_measure(placed: tuple[Project, float]) -> None:
    project, truth = placed
    found = markers.measure_project(project, 30.0)
    assert found is not None
    assert found.mm_per_unit == pytest.approx(truth, rel=0.005)
    assert (found.markers, found.edges, found.photos) == (4, 16, 8)
    assert found.spread < 0.005
    a, b = found.edge  # an edge of a marker on the sheet (y = 0), at the median length
    assert abs(a[1]) < 0.01 and abs(b[1]) < 0.01
    assert 30.0 / np.linalg.norm(np.subtract(b, a)) == pytest.approx(found.mm_per_unit)
    # Printed bigger: the same photos, another scale.
    bigger = markers.measure_project(project, 33.0)
    assert bigger is not None and bigger.mm_per_unit == pytest.approx(truth * 1.1, rel=0.005)


def test_no_markers(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    _succeed(project, "mapping", "m1")
    undistort = _succeed(project, "undistort", "u1")
    marker_scene(undistort / "sparse", undistort / "images")
    blank = Image.new("L", (800, 600), 200)
    for photo in (undistort / "images" / "c").iterdir():
        blank.save(photo, format="PNG")
    assert markers.measure_project(project) is None
    with pytest.raises(markers.MarkerError, match="no camera placement"):
        markers.measure(tmp_path / "nowhere", tmp_path, 30.0)


def test_auto_scale(placed: tuple[Project, float]) -> None:
    project, truth = placed
    note = markers.auto_scale(project)
    assert note is not None and note.startswith("scale from 4 marker(s) of 30 mm in 8 photo(s)")
    current = scale.current(project)
    assert current is not None and current.source == "markers"
    assert current.mm_per_unit == pytest.approx(truth, rel=0.005)
    assert "4 marker(s)" in scale.describe(current)
    assert markers.auto_scale(project) is None  # measured already

    # A scale set by hand on this placement is kept, and compared.
    by_hand = scale.make(((0, 0, 0), (1, 0, 0)), truth * 1.05, "m1")
    scale.save(project, by_hand)
    note = markers.auto_scale(project)
    assert note is not None and "the scale set by hand differs by +5." in note
    assert note.endswith("% and is kept")
    assert scale.current(project) == by_hand


def test_cli(
    placed: tuple[Project, float], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, truth = placed
    assert main(["markers", str(tmp_path / "sheet.svg"), "--size", "25"]) == 0
    assert "should measure 25 mm" in (tmp_path / "sheet.svg").read_text()
    assert main(["scale", str(project.root), "--markers", "--marker-size", "33"]) == 0
    out = capsys.readouterr().out
    assert "scale: from 4 marker(s) of 33 mm" in out
    reopened = Project.open(project.root)
    assert reopened.settings[markers.SIZE_SETTING] == 33.0
    current = scale.current(reopened)
    assert current is not None and current.mm_per_unit == pytest.approx(truth * 1.1, rel=0.005)
    # Correcting the distance needs points picked by hand, not the markers' edge.
    assert main(["scale", str(project.root), "--distance", "50"]) == 1
    assert "pick two points first" in capsys.readouterr().err


def test_detector_freed_before_its_families() -> None:
    # pupil-apriltags frees the families first, then the detector reads them:
    # a use after free at exit that sometimes aborted the process.
    freed: list[str] = []

    class Call:
        def __init__(self, name: str) -> None:
            self.name = name
            self.restype: object = "int"

        def __call__(self, pointer: object) -> None:
            freed.append(f"{self.name}({pointer})")

    class Libc:
        def __getattr__(self, name: str) -> Call:
            call = Call(name)
            setattr(self, name, call)
            return call

    class Detector:
        tag_detector_ptr: object = "detector"
        tag_families = {"tag36h11": "family"}
        libc = Libc()

    detector = Detector()
    markers._destroy(detector)
    markers._destroy(detector)  # already freed: nothing more
    assert freed == ["apriltag_detector_destroy(detector)", "tag36h11_destroy(family)"]
    assert detector.tag_detector_ptr is None


def test_real_detector_frees_cleanly() -> None:
    detector = type(markers._detector())(families=markers.FAMILY)
    assert detector.detect(np.full((64, 64), 255, np.uint8)) == []
    del detector  # with the library's own order this corrupted the heap
