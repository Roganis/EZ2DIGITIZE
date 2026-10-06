# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugins: the manifest, installing, licenses, and running in the pipeline."""

import json
import shutil
import stat
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from scripts import FAKE_BRUSH, FAKE_POSES, make_plugin

from ez2digitize import cli, licenses, pipeline, plugins
from ez2digitize.backends import colmap
from ez2digitize.backends.colmap_model import read_cameras, read_images
from ez2digitize.core.capture import import_files, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import Progress
from ez2digitize.core.stage import load_manifest
from ez2digitize.pipeline import Notice, PipelineError, StageOutput, StageStarted, Tools
from ez2digitize.plugins import PluginError

POSES_COMMAND = [
    "bin/run", "--images", "{captures}", "--list", "{image_list}", "--output", "{output}",
]  # fmt: skip
SPLATS_COMMAND = [
    "bin/run", "{dataset}", "--total-steps", "{steps}",
    "--export-path", "{output}", "--export-name", "splat.ply",
]  # fmt: skip


def _poses(folder: Path, **kwargs: object) -> Path:
    return make_plugin(
        folder,
        provides="poses",
        script=FAKE_POSES,
        command=POSES_COMMAND,
        **kwargs,  # type: ignore[arg-type]
    )


def _splats(folder: Path, **kwargs: object) -> Path:
    return make_plugin(
        folder,
        provides="splats",
        script=FAKE_BRUSH,
        command=SPLATS_COMMAND,
        **kwargs,  # type: ignore[arg-type]
    )


def _use(source: Path) -> plugins.Plugin:
    plugin = plugins.install(source)
    plugins.accept(plugin)
    plugins.choose(plugin.slot, plugin)
    return plugin


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    files = []
    for n, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        Image.new("RGB", (8, 6), (n * 40, 0, 0)).save(tmp_path / name)
        files.append(tmp_path / name)
    import_files(project, files, source="folder")
    return project


# --- the manifest ---------------------------------------------------------------------


def test_read_plugin(tmp_path: Path) -> None:
    folder = _poses(
        tmp_path / "vggt",
        licenses=(("code", "Apache-2.0"), ("model weights", "CC-BY-NC-4.0")),
        extra='gpu = true\nplatforms = ["linux", "macos"]\nwith_masks = ["--masks", "{masks}"]\n'
        '[progress]\npattern = "placing (?P<done>\\\\d+) of (?P<total>\\\\d+)"',
    )
    plugin = plugins.read_plugin(folder)
    assert (plugin.id, plugin.slot, plugin.version, plugin.gpu) == ("vggt", "poses", "1.0", True)
    assert plugin.platforms == ("linux", "darwin")
    assert plugin.backend.name == "plugin:vggt"
    assert [lic.free for lic in plugin.licenses] == [True, False]
    assert not plugin.free
    assert plugin.license_summary() == "Apache-2.0 (code), CC-BY-NC-4.0 (model weights)"
    assert "not known to be a free license" in plugins.describe(plugin)
    parse = plugins.PluginProgress(plugin)
    assert parse("placing 1 of 4") == Progress("Running Test poses", 0.25)
    assert parse("loading") is None


def test_every_plugin_states_its_licenses(tmp_path: Path) -> None:
    with pytest.raises(PluginError, match=r"at least one \[\[license\]\]"):
        plugins.read_plugin(_poses(tmp_path / "p", licenses=()))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("api = 1", "api = 2"), "plugin API 2"),
        (('id = "p"', 'id = "Bad Id"'), "lower case"),
        (('provides = "poses"', 'provides = "meshes"'), "'provides' must be one of"),
        (("'{captures}'", "'{photos}'"), r"unknown placeholder \{photos\}"),
        (("'{captures}'", "'{masks}'"), "goes in 'with_masks'"),
        (("'{captures}'", "'{captures'"), "write {{"),
        (('file = "LICENSE-0.txt"', 'file = "../x.txt"'), "outside the plugin's folder"),
        (('file = "LICENSE-0.txt"', 'file = "GONE.txt"'), "GONE.txt is missing"),
        (("command = [", "platforms = ['amiga']\ncommand = ["), "platform 'amiga'"),
        (("command = [", "gpu = 'yes'\ncommand = ["), "'gpu' must be true or false"),
    ],
)
def test_manifest_rules(tmp_path: Path, change: tuple[str, str], message: str) -> None:
    folder = _poses(tmp_path / "p")
    manifest = folder / plugins.MANIFEST
    text = manifest.read_text()
    assert change[0] in text
    manifest.write_text(text.replace(change[0], change[1], 1))
    with pytest.raises(PluginError, match=message):
        plugins.read_plugin(folder)


def test_not_a_plugin(tmp_path: Path) -> None:
    with pytest.raises(PluginError, match="not a plugin"):
        plugins.read_plugin(tmp_path)


def test_splat_plugins_have_their_own_placeholders(tmp_path: Path) -> None:
    folder = make_plugin(
        tmp_path / "s", provides="splats", script="", command=["bin/run", "{image_list}"]
    )
    with pytest.raises(PluginError, match=r"unknown placeholder \{image_list\}"):
        plugins.read_plugin(folder)


@pytest.mark.parametrize(
    ("expression", "free"),
    [
        ("MIT", True),
        ("Apache-2.0 AND MIT", True),
        ("CC-BY-NC-4.0", False),
        ("Apache-2.0 AND CC-BY-NC-4.0", False),
        ("CC-BY-NC-4.0 OR MIT", True),
        ("(MIT OR Apache-2.0) AND BSD-3-Clause", True),
        ("GPL-2.0-or-later WITH Classpath-exception-2.0", True),
        ("LicenseRef-Inria-Research", False),
    ],
)
def test_free_licenses(expression: str, free: bool) -> None:
    assert plugins.license_is_free(expression) is free


# --- installing, accepting, choosing ---------------------------------------------------


def test_install_accept_choose_remove(tmp_path: Path) -> None:
    source = _poses(tmp_path / "src" / "poses-test")
    plugin = plugins.install(source)
    assert plugin.folder == plugins.plugins_dir() / "poses-test"
    assert [p.id for p in plugins.installed().plugins] == ["poses-test"]
    with pytest.raises(PluginError, match="already installed"):
        plugins.install(source)

    # Not before its licenses are accepted.
    assert not plugins.accepted(plugin)
    with pytest.raises(PluginError, match="accept"):
        plugins.choose("poses", plugin)
    with pytest.raises(PluginError, match="provides poses"):
        plugins.choose("splats", plugin)
    plugins.accept(plugin)
    plugins.choose("poses", plugin)
    assert plugins.chosen("poses") == plugin
    assert plugins.chosen("splats") is None

    # An update with other license texts asks again.
    (source / "LICENSE-0.txt").write_text("new terms\n")
    plugins.install(source, replace_existing=True)
    with pytest.raises(PluginError, match="licenses changed"):
        plugins.chosen("poses")
    plugins.accept(plugins.read_plugin(plugin.folder))
    assert plugins.chosen("poses") is not None

    report = plugins.report()
    assert report["use"] == {"poses": "poses-test", "splats": None}
    assert report["installed"]["poses-test"]["accepted"] is True

    plugins.remove("poses-test")
    assert plugins.installed().plugins == ()
    assert plugins.chosen_id("poses") is None
    with pytest.raises(PluginError, match="no plugin"):
        plugins.remove("poses-test")


def test_chosen_plugin_gone(tmp_path: Path) -> None:
    plugin = _use(_poses(tmp_path / "src" / "p"))
    shutil.rmtree(plugin.folder)
    with pytest.raises(PluginError, match="can't be used"):
        plugins.chosen("poses")


def test_chosen_plugin_for_another_system(tmp_path: Path) -> None:
    other = next(name for name, value in plugins.PLATFORMS.items() if value != sys.platform)
    _use(_poses(tmp_path / "src" / "p", extra=f'platforms = ["{other}"]'))
    with pytest.raises(PluginError, match="doesn't run on this system"):
        plugins.chosen("poses")


def test_broken_plugins_are_listed_as_problems(tmp_path: Path) -> None:
    plugins.plugins_dir().mkdir(parents=True)
    (plugins.plugins_dir() / "junk").mkdir()
    _poses(plugins.plugins_dir() / "misnamed", plugin_id="other")
    found = plugins.installed()
    assert found.plugins == ()
    assert len(found.problems) == 2
    assert any("its folder must be named so" in p for p in found.problems)


def test_install_from_zip(tmp_path: Path) -> None:
    source = _poses(tmp_path / "src" / "zipped")
    archive = tmp_path / "zipped.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"zipped/{path.relative_to(source).as_posix()}")
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            zf.writestr(info, path.read_bytes())
    plugin = plugins.install(archive)
    assert plugin.id == "zipped" and plugin.manifest.is_file()
    if sys.platform != "win32":
        assert (plugin.folder / "bin" / "run").stat().st_mode & stat.S_IXUSR

    for name in ("../escape.txt", "..\\escape.txt", "C:/escape.txt"):
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr(name, "x")
        with pytest.raises(PluginError, match="unsafe path"):
            plugins.install(evil)
    assert not (tmp_path / "escape.txt").exists()

    flat = tmp_path / "flat.zip"
    with zipfile.ZipFile(flat, "w") as zf:
        zf.writestr("README", "no manifest")
    with pytest.raises(PluginError, match="no ez2d-plugin.toml"):
        plugins.install(flat)


# --- in the pipeline ------------------------------------------------------------------


def test_camera_placement_plugin(project: Project, fake_tools: Tools, tmp_path: Path) -> None:
    plugin = _use(
        _poses(
            tmp_path / "src" / "poses-test",
            licenses=(("code", "MIT"), ("model weights", "CC-BY-NC-4.0")),
            extra='with_masks = ["--masks", "{masks}"]\n'
            '[progress]\npattern = "placing (\\\\d+) of (\\\\d+)"\nmessage = "Placing cameras"',
        )
    )
    assert plugins.chosen("poses") == plugin  # what Tools.locate picks up
    tools = replace(fake_tools, poses=plugin)
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_mesh(project, tools, on_event=events.append)

    started = [e.stage for e in events if isinstance(e, StageStarted)]
    assert started[:2] == ["mapping", "undistort"]
    assert "features" not in started and "matching" not in started
    assert result.sparse.registered_images == 3
    manifest = load_manifest(project.stage_dir("mapping"))
    assert manifest is not None and manifest.backend.name == "plugin:poses-test"
    assert manifest.backend.build is not None  # the hash of bin/run
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert notices[0].startswith("placing the cameras with Test poses 1.0 (plugin; MIT (code)")
    assert "not known to be a free license" in notices[0]
    progress = [
        e.event
        for e in events
        if isinstance(e, StageOutput) and e.stage == "mapping" and isinstance(e.event, Progress)
    ]
    assert [p.fraction for p in progress] == [0.5, 1.0]
    assert progress[0].message == "Placing cameras"
    assert "masks:" not in (project.stage_dir("mapping") / "log.txt").read_text()

    # Unchanged: reused. With masks, the plugin gets one for every photo.
    again = pipeline.run_sparse(project, tools)
    assert again.undistorted.run_id == result.sparse.undistorted.run_id
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir(parents=True)
    Image.new("L", (8, 6), 0).save(project.masks_dir / bundle.id / "a.jpg.png")
    pipeline.run_sparse(project, tools)
    log = (project.stage_dir("mapping") / "log.txt").read_text()
    names = " ".join(f"{bundle.id}/{n}.jpg.png" for n in "abc")
    assert f"masks: {names}" in log


def test_pose_priors(project: Project, fake_tools: Tools, tmp_path: Path) -> None:
    """Poses a video's motion track recorded reach the plugin in {priors}."""
    bundle = list_bundles(project)[0]
    pose = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]
    for file in bundle.files[:2]:
        file.metadata["motion"] = {"down": [0, 1, 0], "camera_to_world": pose}
    bundle.save()
    names = [f"{bundle.id}/{n}" for n in ("a.jpg", "b.jpg", "c.jpg")]
    priors = plugins.pose_priors(list_bundles(project), names)
    assert priors == {
        "version": 1,
        "images": {
            name: {"camera_to_world": pose, "frame": bundle.id, "metric": False}
            for name in names[:2]
        },
    }
    source = make_plugin(
        tmp_path / "src" / "poses-test",
        provides="poses",
        script=FAKE_POSES,
        command=[*POSES_COMMAND, "{priors}"],
    )
    plugin = _use(source)
    pipeline.run_mesh(project, replace(fake_tools, poses=plugin))
    written = json.loads((project.stage_dir("mapping") / "priors.json").read_text())
    assert written == priors
    log = (project.stage_dir("mapping") / "log.txt").read_text()
    assert "priors.json" in log  # in the command line


def test_camera_placement_plugin_refined_by_colmap(
    project: Project, fake_tools: Tools, tmp_path: Path
) -> None:
    """refine_poses: the plugin as "poses", then COLMAP's features, pairs and refinement."""
    plugin = _use(_poses(tmp_path / "src" / "poses-test"))
    tools = replace(fake_tools, poses=plugin)
    settings = replace(pipeline.MeshSettings(), refine_poses=True)
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_sparse(project, tools, settings, on_event=events.append)

    started = [e.stage for e in events if isinstance(e, StageStarted)]
    assert started == [
        "poses", "features", "matching", "triangulation", "pose-check", "mapping", "undistort"
    ]  # fmt: skip
    assert result.registered_images == 3
    assert result.model == project.stage_dir("mapping") / "sparse" / "0"
    # The fake plugin puts every camera in one spot looking one way: all pairs.
    pairs = (project.stage_dir("matching") / "pairs.txt").read_text().splitlines()
    assert len(pairs) == 3
    log = (project.stage_dir("matching") / "log.txt").read_text()
    assert "matches_importer" in log and "matching 3 pairs" in log
    # The model point_triangulator starts from mirrors the database.
    known = project.stage_dir("triangulation") / "known_poses"
    assert sorted(read_images(known)) == sorted(read_images(result.model))
    assert {c.model for c in read_cameras(known).values()} == {"SIMPLE_RADIAL"}
    mapping = load_manifest(project.stage_dir("mapping"))
    assert mapping is not None and "--input_path" in mapping.command
    # The coverage notes may use the matches: they came from this placement.
    assert colmap.matched_database(project) is not None
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert "refining its camera placement with COLMAP" in notices

    # Without refining, the plugin is the mapping stage again.
    events.clear()
    pipeline.run_sparse(project, tools, on_event=events.append)
    assert [e.stage for e in events if isinstance(e, StageStarted)][:2] == ["mapping", "undistort"]


def test_camera_placement_plugin_without_masks_says_so(
    project: Project, fake_tools: Tools, tmp_path: Path
) -> None:
    plugin = _use(_poses(tmp_path / "src" / "p"))
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir(parents=True)
    Image.new("L", (8, 6), 0).save(project.masks_dir / bundle.id / "a.jpg.png")
    events: list[pipeline.PipelineEvent] = []
    pipeline.run_sparse(project, replace(fake_tools, poses=plugin), on_event=events.append)
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert "Test poses doesn't use masks; the dense cloud still does" in notices
    started = [e.stage for e in events if isinstance(e, StageStarted)]
    assert started == ["mapping", "undistort", "mask-undistort"]


def test_camera_placement_plugin_failures(
    project: Project, fake_tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _use(_poses(tmp_path / "src" / "p"))
    tools = replace(fake_tools, poses=plugin)
    monkeypatch.setenv("FAKE_FAIL", "plugin")
    with pytest.raises(pipeline.StageFailed, match="stage mapping failed"):
        pipeline.run_sparse(project, tools)
    monkeypatch.delenv("FAKE_FAIL")

    # Wrote no model.
    quiet = make_plugin(
        tmp_path / "src" / "quiet", provides="poses", script="print('done')", command=["bin/run"]
    )
    with pytest.raises(PipelineError, match="Test poses placed no cameras"):
        pipeline.run_sparse(project, replace(fake_tools, poses=plugins.read_plugin(quiet)))

    # Its program is gone.
    missing = make_plugin(
        tmp_path / "src" / "missing", provides="poses", script="", command=["bin/nothing"]
    )
    with pytest.raises(PipelineError, match="can't run 'bin/nothing'"):
        pipeline.run_sparse(project, replace(fake_tools, poses=plugins.read_plugin(missing)))


def test_splat_plugin(
    project: Project, fake_tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _use(_splats(tmp_path / "src" / "splats-test"))
    tools = replace(fake_tools, splats=plugin)  # no Brush, and no GPU needed (gpu = false)
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_splat(project, tools, on_event=events.append)
    assert result.file.read_bytes().startswith(b"ply\n")
    assert [f.suffix for f in result.exports] == [".ply", ".spz"]
    assert result.splat.backend.name == "plugin:splats-test"
    assert result.splat.parameters["total_steps"] == 30_000
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any(n.startswith("training splats with Test splats 1.0 (plugin; MIT") for n in notices)
    assert "gpu" not in result.splat.host

    gpu_plugin = plugins.read_plugin(_splats(tmp_path / "src" / "gpu", extra="gpu = true"))
    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [])
    with pytest.raises(PipelineError, match="no GPU"):
        pipeline.run_splat(project, replace(fake_tools, splats=gpu_plugin))


def test_example_plugin(project: Project, fake_tools: Tools, tmp_path: Path) -> None:
    """tools/plugins/example-poses, with COLMAP's stand-in and this Python."""
    example = Path(__file__).parents[1] / "tools" / "plugins" / "example-poses"
    source = tmp_path / "example-poses"
    shutil.copytree(example, source)
    manifest = source / plugins.MANIFEST
    manifest.write_text(manifest.read_text().replace('"python3"', repr(sys.executable)))
    plugin = _use(source)
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_sparse(project, replace(fake_tools, poses=plugin), on_event=events.append)
    assert result.registered_images == 3
    progress = [
        e.event.fraction
        for e in events
        if isinstance(e, StageOutput) and isinstance(e.event, Progress) and e.stage == "mapping"
    ]
    assert progress == [pytest.approx(1 / 3, abs=1e-3), pytest.approx(2 / 3, abs=1e-3), 1.0]


# --- the command line -----------------------------------------------------------------


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _poses(tmp_path / "src" / "vggt", licenses=(("model weights", "CC-BY-NC-4.0"),))
    assert cli.main(["plugins"]) == 0
    assert "Camera placement (poses): COLMAP" in capsys.readouterr().out

    assert cli.main(["plugins", "install", str(source)]) == 0
    out = capsys.readouterr().out
    assert "installed Test poses 1.0 (poses)" in out
    assert "ez2d plugins accept vggt" in out  # not a terminal: never accepted silently
    assert cli.main(["plugins", "use", "poses", "vggt"]) == 1
    assert "accept Test poses's licenses" in capsys.readouterr().err

    assert cli.main(["plugins", "license", "vggt"]) == 0
    out = capsys.readouterr().out
    assert "model weights, CC-BY-NC-4.0  (not known to be a free license" in out
    assert "CC-BY-NC-4.0 terms for the model weights" in out
    assert cli.main(["plugins", "accept", "vggt"]) == 0
    assert cli.main(["plugins", "use", "poses", "vggt"]) == 0
    capsys.readouterr()
    assert cli.main(["plugins", "list"]) == 0
    out = capsys.readouterr().out
    assert "Camera placement (poses): vggt" in out and "licenses accepted" in out
    assert "Test poses 1.0: CC-BY-NC-4.0 (model weights)" in licenses.summary()

    assert cli.main(["plugins", "use", "poses", "built-in"]) == 0
    assert plugins.chosen("poses") is None
    assert cli.main(["plugins", "remove", "vggt"]) == 0
    assert cli.main(["plugins", "accept", "vggt"]) == 1
    assert "no plugin 'vggt' is installed" in capsys.readouterr().err

    # --accept, for scripts.
    assert cli.main(["plugins", "install", str(source), "--accept"]) == 0
    assert plugins.accepted(plugins.read_plugin(plugins.plugins_dir() / "vggt"))
