# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

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


def test_locate(tmp_path: Path, fake_tool: FakeTool, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = fake_tool(tmp_path / "colmap", "COLMAP 4.2.1 (Commit abc without CUDA)")
    found = colmap.locate(exe)
    assert found == Colmap(exe, "4.2.1") and found.supported
    monkeypatch.setenv("EZ2D_COLMAP", str(fake_tool(tmp_path / "old", "COLMAP 3.9.1 -- SfM")))
    old = colmap.locate()
    assert old.version == "3.9.1" and not old.supported


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
    Image.new("RGB", (6, 4)).save(bundle.root / "b.jpg")  # a real image, 6x4 raw
    masks = project.masks_dir
    (masks / bundle.id).mkdir()
    (masks / bundle.id / "a.jpg.png").write_bytes(b"m")
    spec = colmap.extract_features(TOOL, project, [bundle], masks=masks)

    stage_dir = project.stage_dir("features")
    assert spec.argv[:2] == [TOOL.path, "feature_extractor"]
    assert _opt(spec, "--database_path") == str(stage_dir / "database.db")
    assert _opt(spec, "--image_path") == str(project.captures_dir)
    assert _opt(spec, "--ImageReader.mask_path") == str(stage_dir / "masks")
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
    # COLMAP skips an image without a mask: b.jpg gets a white one of its size.
    staged = stage_dir / "masks" / bundle.id
    assert (staged / "a.jpg.png").read_bytes() == b"m"
    with Image.open(staged / "b.jpg.png") as white:
        assert (white.mode, white.size, white.getextrema()) == ("L", (6, 4), (255, 255))


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
    assert "--SequentialMatching.vocab_tree_path" not in seq.argv


def test_matching_for_large_sets(project: Project, tmp_path: Path) -> None:
    features = _manifest("features")
    tree = tmp_path / "tree.bin"
    loops = colmap.match_features(
        TOOL, project, features, options=MatchOptions("sequential", vocab_tree=tree)
    )
    assert _opt(loops, "--SequentialMatching.loop_detection") == "1"
    assert _opt(loops, "--SequentialMatching.vocab_tree_path") == str(tree)
    # The tree is recorded by its hash, not where it is.
    assert loops.parameters["vocab_tree"] == colmap.VOCAB_TREES["sift"].sha256
    moved = colmap.match_features(
        TOOL, project, features, options=MatchOptions("sequential", vocab_tree=tmp_path / "x")
    )
    assert moved.cache_key() == loops.cache_key()

    spatial = colmap.match_features(TOOL, project, features, options=MatchOptions("spatial"))
    assert spatial.argv[1] == "spatial_matcher"
    assert _opt(spatial, "--SpatialMatching.max_num_neighbors") == "50"
    assert _opt(spatial, "--SpatialMatching.ignore_z") == "1"

    similar = colmap.match_features(
        TOOL, project, features, options=MatchOptions("vocab_tree", vocab_tree=tree)
    )
    assert similar.argv[1] == "vocab_tree_matcher"
    assert _opt(similar, "--VocabTreeMatching.vocab_tree_path") == str(tree)
    assert _opt(similar, "--VocabTreeMatching.num_images") == "100"
    with pytest.raises(BackendError, match="needs the tree"):
        colmap.match_features(TOOL, project, features, options=MatchOptions("vocab_tree"))


def test_learned_features(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg")
    sift = colmap.extract_features(TOOL, project, [bundle])
    assert _opt(sift, "--SiftExtraction.max_num_features") == "8192"
    assert "--FeatureExtraction.type" not in sift.argv
    model = tmp_path / "aliked.onnx"
    aliked = colmap.extract_features(
        TOOL, project, [bundle], options=FeatureOptions(kind="aliked", model=model)
    )
    assert _opt(aliked, "--FeatureExtraction.type") == "ALIKED_N16ROT"
    assert _opt(aliked, "--AlikedExtraction.n16rot_model_path") == str(model)
    assert _opt(aliked, "--AlikedExtraction.max_num_features") == "2048"
    assert "--SiftExtraction.max_num_features" not in aliked.argv
    assert aliked.parameters["model"] == colmap.ALIKED_MODEL.sha256
    elsewhere = colmap.extract_features(
        TOOL, project, [bundle], options=FeatureOptions(kind="aliked", model=tmp_path / "x")
    )
    assert elsewhere.cache_key() == aliked.cache_key() != sift.cache_key()
    with pytest.raises(BackendError, match="ALIKED model"):
        colmap.extract_features(TOOL, project, [bundle], options=FeatureOptions(kind="aliked"))

    features = _manifest("features")
    glue = tmp_path / "lightglue.onnx"
    matched = colmap.match_features(
        TOOL, project, features, options=MatchOptions(features="aliked", lightglue=glue)
    )
    assert _opt(matched, "--FeatureMatching.type") == "ALIKED_LIGHTGLUE"
    assert _opt(matched, "--AlikedMatching.lightglue_model_path") == str(glue)
    assert matched.parameters["lightglue"] == colmap.LIGHTGLUE_MODEL.sha256
    assert "--FeatureMatching.type" not in colmap.match_features(TOOL, project, features).argv
    with pytest.raises(BackendError, match="LightGlue"):
        colmap.match_features(TOOL, project, features, options=MatchOptions(features="aliked"))


def test_pinned_download(server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from ez2digitize.core import download

    monkeypatch.setenv("EZ2D_MODELS_DIR", str(tmp_path / "models"))
    for var in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    data = b"tree" * 1000
    (tmp_path / "www" / "tree.bin").write_bytes(data)
    pinned = colmap.Pinned(
        "tree.bin", f"{server}/tree.bin", hashlib.sha256(data).hexdigest(), len(data)
    )
    assert pinned.path == tmp_path / "models" / "tree.bin"
    assert colmap.find_pinned(pinned) is None
    # What fetch_pinned does (the tests stand it in, to stay offline).
    path = download.fetch(pinned.url, pinned.path, pinned.sha256, size=pinned.size)
    assert colmap.find_pinned(pinned) == path and path.read_bytes() == data
    path.write_bytes(data[:-1])  # cut short: not the pinned file
    assert colmap.find_pinned(pinned) is None


def test_pinned_files_are_colmaps() -> None:
    """The URLs and hashes COLMAP 4.2.1's source pins (retrieval/resources.h,
    feature/resources.h); checked against the downloaded files."""
    pinned = [*colmap.VOCAB_TREES.values(), colmap.ALIKED_MODEL, colmap.LIGHTGLUE_MODEL]
    for file in pinned:
        assert file.url.startswith("https://github.com/colmap/colmap/releases/download/")
        assert file.url.endswith("/" + file.name) and len(file.sha256) == 64


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


def _per_image_database(path: Path, count: int) -> None:
    """The tables COLMAP 4.2.1 writes for `single_camera_per_image` extraction."""
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                                  height INTEGER, params BLOB, prior_focal_length INTEGER);
            CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER,
                               ref_sensor_type INTEGER);
            CREATE TABLE rig_sensors (rig_id INTEGER, sensor_id INTEGER, sensor_type INTEGER,
                                      sensor_from_rig BLOB);
            CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
            CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER, sensor_id INTEGER,
                                     sensor_type INTEGER);
            CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
            """
        )
        for i in range(1, count + 1):
            db.execute("INSERT INTO cameras VALUES (?, 2, 4272, 2848, x'00', 1)", (i,))
            db.execute("INSERT INTO rigs VALUES (?, ?, 0)", (i, i))
            db.execute("INSERT INTO frames VALUES (?, ?)", (i, i))
            db.execute("INSERT INTO frame_data VALUES (?, ?, ?, 0)", (i, i, i))
            db.execute("INSERT INTO images VALUES (?, ?, ?)", (i, f"c/{i}.jpg", i))


def test_merge_cameras(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _per_image_database(database, 5)
    groups = {"c/1.jpg": "wide", "c/2.jpg": "zoom", "c/3.jpg": "wide", "c/4.jpg": "zoom"}
    assert colmap.merge_cameras(database, groups) == 2
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT camera_id FROM cameras").fetchall() == [(1,), (2,), (5,)]
        assert db.execute("SELECT name, camera_id FROM images").fetchall() == [
            ("c/1.jpg", 1), ("c/2.jpg", 2), ("c/3.jpg", 1), ("c/4.jpg", 2), ("c/5.jpg", 5),
        ]  # fmt: skip
        # As single-camera extraction writes it: frames of the kept rig.
        assert db.execute("SELECT rig_id FROM rigs").fetchall() == [(1,), (2,), (5,)]
        assert db.execute("SELECT frame_id, rig_id FROM frames").fetchall() == [
            (1, 1), (2, 2), (3, 1), (4, 2), (5, 5),
        ]  # fmt: skip
        assert db.execute("SELECT data_id, sensor_id FROM frame_data").fetchall() == [
            (1, 1), (2, 2), (3, 1), (4, 2), (5, 5),
        ]  # fmt: skip


def test_matching_merges_camera_groups(project: Project, tmp_path: Path) -> None:
    features = _manifest("features")
    project.stage_dir("features").mkdir(parents=True)
    _per_image_database(project.stage_dir("features") / colmap.DATABASE, 2)
    plain = colmap.match_features(TOOL, project, features)
    spec = colmap.match_features(
        TOOL, project, features, camera_groups={"c/1.jpg": "a", "c/2.jpg": "a"}
    )
    assert spec.cache_key() != plain.cache_key()
    folder = project.stage_dir("matching")
    folder.mkdir(parents=True)
    assert spec.prepare is not None
    spec.prepare(folder)
    with sqlite3.connect(folder / colmap.DATABASE) as db:
        assert db.execute("SELECT count(*) FROM cameras").fetchone() == (1,)
    with sqlite3.connect(project.stage_dir("features") / colmap.DATABASE) as db:
        assert db.execute("SELECT count(*) FROM cameras").fetchone() == (2,)  # untouched


def test_per_image_grouping_flag(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path, "a.jpg")
    options = colmap.FeatureOptions(camera_grouping="per_image")
    spec = colmap.extract_features(TOOL, project, [bundle], options=options)
    assert _opt(spec, "--ImageReader.single_camera_per_image") == "1"
    assert _opt(spec, "--ImageReader.single_camera_per_folder") == "0"
