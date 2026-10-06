# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ez2digitize import masks
from ez2digitize.core.capture import CaptureBundle, CaptureError, import_files
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Progress


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _bundle(project: Project, folder: Path, *names: str) -> CaptureBundle:
    folder.mkdir(parents=True, exist_ok=True)
    for n, name in enumerate(names):
        Image.new("RGB", (4, 4), (n * 30, 0, 0)).save(folder / name)
    return import_files(project, [folder / n for n in names], source="folder")


def _states(project: Project, bundles: list[CaptureBundle]) -> dict[str, str]:
    return {e.name: e.state for e in masks.review(project, bundles)}


def test_make_masks_puts_them_in_use(
    project: Project, tmp_path: Path, fake_mask_worker: Path
) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.jpg")
    events: list[object] = []
    runs = masks.make_masks(project, [bundle], fake_mask_worker, on_event=events.append)

    assert [(r.capture, r.reused, r.added, r.dropped) for r in runs] == [(bundle.id, False, 2, 0)]
    assert (project.masks_dir / bundle.id / "a.jpg.png").is_file()
    assert _states(project, [bundle]) == {"a.jpg": "automatic", "b.jpg": "automatic"}
    assert masks.has_masks(project)
    fractions = [e.fraction for e in events if isinstance(e, Progress)]
    assert fractions == [0.0, 0.0, 0.5]

    # Same photos: the stage is reused, nothing is added twice.
    again = masks.make_masks(project, [bundle], fake_mask_worker)
    assert (again[0].reused, again[0].added) == (True, 0)


def test_leaving_photos_out_does_not_mask_again(
    project: Project, tmp_path: Path, fake_mask_worker: Path
) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.jpg")
    masks.make_masks(project, [bundle], fake_mask_worker)
    bundle.set_excluded(["b.jpg"])
    assert masks.make_masks(project, [bundle], fake_mask_worker)[0].reused


def test_new_capture_is_masked_on_its_own(
    project: Project, tmp_path: Path, fake_mask_worker: Path
) -> None:
    first = _bundle(project, tmp_path / "one", "a.jpg")
    masks.make_masks(project, [first], fake_mask_worker)
    second = _bundle(project, tmp_path / "two", "c.jpg")
    runs = masks.make_masks(project, [first, second], fake_mask_worker)
    assert [r.reused for r in runs] == [True, False]


def test_empty_masks_start_dropped(
    project: Project, tmp_path: Path, fake_mask_worker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_COVERAGE", "a.jpg=0")
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.jpg")
    runs = masks.make_masks(project, [bundle], fake_mask_worker)
    assert (runs[0].added, runs[0].dropped) == (2, 1)
    entries = {e.name: e for e in masks.review(project, [bundle])}
    assert entries["a.jpg"].state == "dropped"
    assert entries["a.jpg"].flags == ["nothing found"]
    assert entries["a.jpg"].mask == project.masks_dir / bundle.id / "dropped" / "a.jpg.png"
    assert not (project.masks_dir / bundle.id / "a.jpg.png").exists()


def test_drop_restore_and_clear(project: Project, tmp_path: Path, fake_mask_worker: Path) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.jpg")
    masks.make_masks(project, [bundle], fake_mask_worker)

    masks.drop(project, bundle, ["a.jpg"])
    assert _states(project, [bundle]) == {"a.jpg": "dropped", "b.jpg": "automatic"}
    # Masks made again (a newer worker) keep the review decisions.
    masks.make_masks(project, [bundle], fake_mask_worker, force=True)
    assert _states(project, [bundle]) == {"a.jpg": "dropped", "b.jpg": "automatic"}
    masks.restore(project, bundle, ["a.jpg"])
    assert _states(project, [bundle])["a.jpg"] == "automatic"

    masks.drop(project, bundle, ["a.jpg", "b.jpg"])
    assert not masks.has_masks(project)  # dropped ones don't count
    assert masks.clear_auto(project, bundle) == 2
    assert _states(project, [bundle]) == {"a.jpg": "none", "b.jpg": "none"}
    assert not (project.masks_dir / bundle.id).exists()
    with pytest.raises(CaptureError, match="has no mask"):
        masks.drop(project, bundle, ["a.jpg"])


def test_imported_masks_win(project: Project, tmp_path: Path, fake_mask_worker: Path) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.jpg")
    imported = tmp_path / "masks"
    imported.mkdir()
    Image.new("L", (4, 4), 128).save(imported / "a.jpg.png")
    assert masks.import_folder_masks(project, bundle, imported) == ["a.jpg"]

    runs = masks.make_masks(project, [bundle], fake_mask_worker)
    assert runs[0].added == 1
    assert _states(project, [bundle]) == {"a.jpg": "imported", "b.jpg": "automatic"}
    with Image.open(project.masks_dir / bundle.id / "a.jpg.png") as kept:
        assert kept.getextrema() == (128, 128)

    # Importing over an automatic (even dropped) mask replaces it.
    masks.drop(project, bundle, ["b.jpg"])
    Image.new("L", (4, 4), 128).save(imported / "b.jpg.png")
    masks.import_folder_masks(project, bundle, imported)
    assert _states(project, [bundle]) == {"a.jpg": "imported", "b.jpg": "imported"}
    assert masks.clear_auto(project, bundle) == 0


def test_worker_failure_and_cancel(
    project: Project, tmp_path: Path, fake_mask_worker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg")
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(masks.MaskingError, match="failed .exit code 1"):
        masks.make_masks(project, [bundle], fake_mask_worker)
    assert not masks.has_masks(project)
    monkeypatch.delenv("FAKE_FAIL")
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(masks.MaskingCancelled):
        masks.make_masks(project, [bundle], fake_mask_worker, cancel=cancel)


def test_mask_spec(project: Project, tmp_path: Path) -> None:
    bundle = _bundle(project, tmp_path / "in", "a.jpg", "b.png")
    (bundle.root / "c.heic").write_bytes(b"heic")  # COLMAP can't read it: no mask
    bundle.files.append(
        bundle.files[0].__class__("c.heic", "c.heic", "image", 4, "x", excluded=True)
    )
    bundle.set_excluded(["b.png"])
    spec = masks.mask_spec(project, bundle, tmp_path / "model.onnx")
    assert spec.name == f"masks-{bundle.id}"
    assert spec.inputs["model"] == f"sha256:{masks.MODEL.sha256}"
    assert spec.parameters == {"model": "isnet-general-use", "masker_version": 1}

    stage_dir = project.stage_dir(spec.name)
    stage_dir.mkdir(parents=True)
    assert spec.prepare is not None
    spec.prepare(stage_dir)
    jobs = json.loads((stage_dir / "jobs.json").read_text())["jobs"]
    assert [j["key"] for j in jobs] == ["a.jpg", "b.png"]  # left-out photos too
    assert jobs[0]["mask"] == str(stage_dir / "masks" / bundle.id / "a.jpg.png")

    # Other models, or a newer worker, make the masks again.
    other = masks.MaskModel("other", "u", "0" * 64, 1, "MIT", 512)
    assert masks.mask_spec(project, bundle, tmp_path / "m", model=other).cache_key() != (
        spec.cache_key()
    )


def test_flags() -> None:
    assert masks.flags({"coverage": 0.001}, None) == ["nothing found"]
    assert masks.flags({"coverage": 0.95}, None) == ["almost the whole photo"]
    assert masks.flags({"coverage": 0.2, "uncertain": 0.1}, None) == ["unsure"]
    assert masks.flags({"coverage": 0.05}, 0.25) == ["unlike the others"]
    assert masks.flags({"coverage": 0.2, "uncertain": 0.01}, 0.25) == []


def test_progress_parser() -> None:
    parse = masks.MaskProgress()
    assert parse("model isnet.onnx loaded, 62 images") == Progress("loading the masking model", 0.0)
    assert parse("mask 3/4 IMG_3.JPG") == Progress("masking IMG_3.JPG (3/4)", 0.5)
    assert parse("done") is None


def test_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EZ2D_MODELS_DIR", str(tmp_path))
    assert masks.model_file() == tmp_path / "isnet-general-use.onnx"
    assert masks.find_model() is None
    monkeypatch.delenv("EZ2D_MODELS_DIR")
    variable = {"darwin": None, "win32": "LOCALAPPDATA"}.get(sys.platform, "XDG_CACHE_HOME")
    if variable:
        monkeypatch.setenv(variable, str(tmp_path / "cache"))
        assert masks.models_dir() == tmp_path / "cache" / "ez2digitize" / "models"


def test_download_model(server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EZ2D_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    data = bytes(range(256)) * 9000  # a bit over 2 MB: several chunks
    (tmp_path / "www" / "m.onnx").write_bytes(data)
    model = masks.MaskModel(
        "tiny", f"{server}/m.onnx", hashlib.sha256(data).hexdigest(), len(data), "MIT", 8
    )
    seen: list[tuple[int, int]] = []
    path = masks.download_model(model, on_progress=lambda r, t: seen.append((r, t)))
    assert path.read_bytes() == data
    assert masks.find_model(model) == path
    assert seen[-1] == (len(data), len(data)) and len(seen) == 3

    damaged = masks.MaskModel("bad", model.url, "0" * 64, len(data), "MIT", 8)
    with pytest.raises(masks.MaskingError, match="damaged"):
        masks.download_model(damaged)
    assert not list((tmp_path / "models").glob("bad*"))

    missing = masks.MaskModel("gone", f"{server}/nope.onnx", "0" * 64, 1, "MIT", 8)
    with pytest.raises(masks.MaskingError, match="save it as .*gone.onnx"):
        masks.download_model(missing)

    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(masks.MaskingCancelled):
        masks.download_model(
            masks.MaskModel("c", model.url, model.sha256, len(data), "MIT", 8), cancel=cancel
        )
    assert not list((tmp_path / "models").glob("c.*"))


def test_worker_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    assert masks.worker_argv() == [sys.executable, "-m", "ez2digitize.mask_worker"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert masks.worker_argv() == [sys.executable, "mask-worker"]
