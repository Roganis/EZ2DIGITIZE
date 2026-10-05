# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sqlite3
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from ez2digitize.backends import colmap
from ez2digitize.backends.colmap import (
    Colmap,
    ColmapProgress,
    FeatureOptions,
    MapperOptions,
    MatchOptions,
    best_model,
    image_names,
    parse_version,
)
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.core.capture import CaptureBundle, import_files
from ez2digitize.core.project import Project
from ez2digitize.core.runner import Progress
from ez2digitize.core.stage import StageManifest, StageSpec

FakeTool = Callable[[Path, str], Path]
TOOL = Colmap(path=Path("/opt/colmap/bin/colmap"), version="4.2.1")


def _bundle(project: Project, tmp_path: Path, *names: str) -> CaptureBundle:
    files = []
    for name in names:
        path = tmp_path / "in" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files.append(path)
    return import_files(project, files, source="folder")


def _manifest(stage: str, run_id: str = "abc") -> StageManifest:
    return StageManifest(
        stage=stage, run_id=run_id, status="succeeded", cache_key="k",
        backend=TOOL.backend, command=[], parameters={}, inputs={}, started="",
        finished="", wall_s=0, cpu_s=0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip


def _opt(spec: StageSpec, name: str) -> str:
    argv = [str(a) for a in spec.argv]
    return argv[argv.index(name) + 1]


# --- version and discovery -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("COLMAP 4.2.1 (Commit 1a2b3c4 on 2026-06-01 without CUDA)\nUsage:", "4.2.1"),
        ("COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo\n(Commit Unknown)", "3.9.1"),
        ("something else", None),
    ],
)
def test_parse_version(text: str, version: str | None) -> None:
    assert parse_version(text) == version


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executables")
def test_locate(tmp_path: Path, fake_tool: FakeTool, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = fake_tool(tmp_path / "colmap", "COLMAP 4.2.1 (Commit abc without CUDA)")
    found = colmap.locate(exe)
    assert found == Colmap(exe, "4.2.1") and found.supported
    monkeypatch.setenv("EZ2D_COLMAP", str(fake_tool(tmp_path / "old", "COLMAP 3.9.1 -- SfM")))
    old = colmap.locate()
    assert old.version == "3.9.1" and not old.supported


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executables")
def test_locate_errors(tmp_path: Path, fake_tool: FakeTool) -> None:
    with pytest.raises(BackendMissing, match="not found"):
        colmap.locate(tmp_path / "missing")
    with pytest.raises(BackendMissing, match="doesn't look like COLMAP"):
        colmap.locate(fake_tool(tmp_path / "colmap", "hello"))


# --- command builders -------------------------------------------------------------


def test_excluded_images_are_left_out(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg", "b.jpg")
    before = colmap.extract_features(TOOL, project, [bundle]).cache_key()
    bundle.set_excluded(["b.jpg"])
    assert image_names([bundle]) == [f"{bundle.id}/a.jpg"]
    assert colmap.extract_features(TOOL, project, [bundle]).cache_key() != before
    bundle.set_excluded(["b.jpg"], excluded=False)
    assert colmap.extract_features(TOOL, project, [bundle]).cache_key() == before


def test_image_names(project: Project, tmp_path: Path) -> None:
    a = _bundle(project, tmp_path, "b.JPG", "a.png", "clip.mp4")
    assert image_names([a]) == [f"{a.id}/b.JPG", f"{a.id}/a.png"]
    with pytest.raises(BackendError, match="no images"):
        image_names([])
    heic = _bundle(project, tmp_path, "c.heic")
    with pytest.raises(BackendError, match="can't read 1 image"):
        image_names([a, heic])


def test_extract_features(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg", "b.jpg")
    masks = project.masks_dir
    (masks / bundle.id).mkdir()
    (masks / bundle.id / "a.jpg.png").write_bytes(b"m")
    spec = colmap.extract_features(TOOL, project, [bundle], masks=masks)

    stage_dir = project.stage_dir("features")
    assert spec.argv[:2] == [TOOL.path, "feature_extractor"]
    assert _opt(spec, "--database_path") == str(stage_dir / "database.db")
    assert _opt(spec, "--image_path") == str(project.captures_dir)
    assert _opt(spec, "--ImageReader.mask_path") == str(masks)
    assert _opt(spec, "--ImageReader.single_camera_per_folder") == "1"
    assert _opt(spec, "--ImageReader.single_camera") == "0"
    assert _opt(spec, "--FeatureExtraction.use_gpu") == "0"
    assert "--FeatureExtraction.num_threads" not in spec.argv
    assert spec.parameters["masked"] is True
    assert set(spec.inputs) == {"captures", "masks"}

    stage_dir.mkdir(parents=True)
    assert spec.prepare is not None
    spec.prepare(stage_dir)
    assert (stage_dir / "image_list.txt").read_text() == f"{bundle.id}/a.jpg\n{bundle.id}/b.jpg\n"


def test_feature_cache_key_follows_options_and_captures(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg")
    base = colmap.extract_features(TOOL, project, [bundle]).cache_key()
    assert colmap.extract_features(TOOL, project, [bundle]).cache_key() == base
    single = FeatureOptions(camera_grouping="single", threads=4)
    spec = colmap.extract_features(TOOL, project, [bundle], options=single)
    assert spec.cache_key() != base
    assert _opt(spec, "--ImageReader.single_camera") == "1"
    assert _opt(spec, "--FeatureExtraction.num_threads") == "4"
    other = _bundle(project, tmp_path, "b.jpg")
    assert colmap.extract_features(TOOL, project, [bundle, other]).cache_key() != base


def test_match_features_copies_database(project: Project) -> None:
    features_dir = project.stage_dir("features")
    features_dir.mkdir(parents=True)
    with sqlite3.connect(features_dir / "database.db") as db:
        db.execute("create table t (x)")
    db.close()
    spec = colmap.match_features(TOOL, project, _manifest("features", "run1"))
    assert spec.argv[1] == "exhaustive_matcher"
    assert spec.inputs == {"features": "run:run1"}
    assert "--SequentialMatching.overlap" not in spec.argv

    matching_dir = project.stage_dir("matching")
    matching_dir.mkdir()
    assert spec.prepare is not None
    spec.prepare(matching_dir)
    assert (matching_dir / "database.db").read_bytes() == (
        features_dir / "database.db"
    ).read_bytes()

    seq = colmap.match_features(
        TOOL, project, _manifest("features"), options=MatchOptions("sequential", 15)
    )
    assert seq.argv[1] == "sequential_matcher"
    assert _opt(seq, "--SequentialMatching.overlap") == "15"
    assert _opt(seq, "--SequentialMatching.loop_detection") == "0"


@pytest.mark.parametrize(("kind", "command", "threads"), [
    ("incremental", "mapper", "--Mapper.num_threads"),
    ("global", "global_mapper", "--GlobalMapper.num_threads"),
])  # fmt: skip
def test_map_sparse(project: Project, kind: str, command: str, threads: str) -> None:
    options = MapperOptions(kind=kind, threads=2)  # type: ignore[arg-type]
    spec = colmap.map_sparse(TOOL, project, _manifest("matching"), options=options)
    assert spec.argv[1] == command
    assert _opt(spec, threads) == "2"
    assert _opt(spec, "--database_path") == str(project.stage_dir("matching") / "database.db")
    assert _opt(spec, "--output_path") == str(project.stage_dir("mapping") / "sparse")


def test_undistort(project: Project) -> None:
    model = project.stage_dir("mapping") / "sparse" / "1"
    spec = colmap.undistort(TOOL, project, _manifest("mapping"), model=model)
    assert spec.argv[1] == "image_undistorter"
    assert _opt(spec, "--input_path") == str(model)
    assert _opt(spec, "--output_path") == str(project.stage_dir("undistort"))
    assert spec.parameters["model"] == "1"


def test_failed_upstream_is_rejected(project: Project) -> None:
    failed = _manifest("features")
    failed.status = "failed"
    with pytest.raises(ValueError, match="did not succeed"):
        colmap.match_features(TOOL, project, failed)


# --- results ----------------------------------------------------------------------


def _model(folder: Path, registered: int) -> None:
    folder.mkdir(parents=True)
    (folder / "images.bin").write_bytes(struct.pack("<Q", registered) + b"\0" * 16)


def test_best_model(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    assert best_model(sparse) is None
    _model(sparse / "0", 12)
    _model(sparse / "1", 30)
    _model(sparse / "10", 3)
    (sparse / "2").mkdir()  # no model files
    assert [m.name for m in colmap.models(sparse)] == ["0", "1", "10"]
    assert best_model(sparse) == sparse / "1"
    assert colmap.registered_images(sparse / "2") == 0


# --- progress ---------------------------------------------------------------------


def _feed(parser: ColmapProgress, lines: list[str]) -> list[Progress]:
    return [p for line in lines if (p := parser(line)) is not None]


def test_progress_features_and_headings() -> None:
    lines = [
        "I20261005 14:12:18.053894  9857 misc.cc:198] ",
        "==============================================================================",
        "Feature extraction",
        "==============================================================================",
        "I20261005 14:12:21.089553  9866 feature_extraction.cc:254] Processed file [1/32]",
        "I20261005 14:12:21.089715  9866 feature_extraction.cc:257]   Name:   view_001.jpg",
        "I20261005 14:12:32.832480  9866 feature_extraction.cc:254] Processed file [32/32]",
        "I20261005 14:12:32.841760  9857 timer.cc:91] Elapsed time: 0.246 [minutes]",
    ]
    assert _feed(ColmapProgress(), lines) == [
        Progress("Feature extraction"),
        Progress("Extracting features 1/32", 1 / 32),
        Progress("Extracting features 32/32", 1.0),
    ]


def test_progress_matching_and_undistortion() -> None:
    parser = ColmapProgress()
    assert parser(
        "I20261005 14:12:32.938627  9874 pairing.cc:214] Processing block [2/3, 1/3]"
    ) == Progress("Matching block 2/3, 1/3", 4 / 9)
    assert parser("Processing image [5/10]") == Progress("Matching 5/10", 0.5)
    assert parser("Undistorting image [3/4]") == Progress("Undistorting images 3/4", 0.75)


@pytest.mark.parametrize(
    "line",
    [
        "I20261005 14:13:04.1  9893 incremental_mapper.cc:1] Registering image #12 (3)",  # 3.9
        "I20261005 14:13:04.1  9893 incremental_pipeline.cc:620] "
        "Registering image #12 (num_reg_frames=3)",  # 4.2
    ],
)
def test_progress_mapper(line: str) -> None:
    assert ColmapProgress(total_images=12)(line) == Progress("Registered 3/12 images", 0.25)
    assert ColmapProgress()(line) == Progress("Registered 3 images")


def test_progress_global_mapper_headings() -> None:
    lines = ["=" * 78, "Running rotation averaging", "=" * 78, "something else"]
    assert _feed(ColmapProgress(), lines) == [Progress("Running rotation averaging")]


@pytest.mark.parametrize(
    "prefix",
    [
        "I20261005 14:12:21.089553  9866 feature_extraction.cc:271] ",  # Linux
        "I20261005 17:58:01.123456 0x16b8f3000 feature_extraction.cc:271] ",  # macOS
        "I1005 17:58:01.123456 12345 feature_extraction.cc:271] ",  # older glog
    ],
)
def test_progress_glog_prefixes(prefix: str) -> None:
    assert ColmapProgress()(prefix + "Processed file [2/4]") == Progress(
        "Extracting features 2/4", 0.5
    )


@pytest.mark.parametrize(
    ("max_size", "available_gib", "cpus", "threads"),
    [
        (3200, 21, 16, 7),  # the GRE: 21 GiB free, 16 threads -> 7, not 16
        (3200, 64, 16, 16),  # plenty of memory: all cores
        (3200, 6, 8, 2),  # an 8 GB M1
        (3200, 1, 8, 1),  # never below one thread
        (1600, 6, 8, 8),  # quarter the area, quarter the memory per thread
    ],
)
def test_feature_threads(max_size: int, available_gib: int, cpus: int, threads: int) -> None:
    assert colmap.feature_threads(max_size, available_gib * 1024**3, cpus) == threads


def test_thread_count_is_not_part_of_the_cache_key(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg")
    one = colmap.extract_features(TOOL, project, [bundle], options=FeatureOptions(threads=1))
    many = colmap.extract_features(TOOL, project, [bundle], options=FeatureOptions(threads=16))
    assert one.cache_key() == many.cache_key()
    assert "threads" not in one.parameters
    assert _opt(many, "--FeatureExtraction.num_threads") == "16"
