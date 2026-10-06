# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from ez2digitize.core import capture as capture_mod
from ez2digitize.core.capture import (
    CaptureBundle,
    CaptureError,
    classify,
    import_files,
    import_folder,
    list_bundles,
)
from ez2digitize.core.project import Project

NOW = datetime(2026, 10, 5, 20, 32, 0)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_classify() -> None:
    assert classify(Path("a.JPG")) == "image"
    assert classify(Path("a.heic")) == "image"
    assert classify(Path("clip.MOV")) == "video"
    assert classify(Path("notes.txt")) is None


def test_import_files_copies_untouched_and_records_hashes(project: Project, tmp_path: Path) -> None:
    src = _write(tmp_path / "in" / "IMG_0001.jpg", b"\xff\xd8 pretend jpeg")
    bundle = import_files(project, [src], source="upload", device={"model": "Pixel 8"}, now=NOW)

    assert bundle.id == "20261005-203200"
    assert bundle.root == project.captures_dir / bundle.id
    copied = bundle.root / "IMG_0001.jpg"
    assert copied.read_bytes() == src.read_bytes()
    assert copied.stat().st_mtime == pytest.approx(src.stat().st_mtime)
    data = json.loads((bundle.root / "capture.json").read_text())
    assert data["source"] == "upload"
    assert data["device"] == {"model": "Pixel 8"}
    assert data["files"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert data["files"][0]["size"] == len(src.read_bytes())
    assert bundle.images == [copied]


def test_load_round_trip(project: Project, tmp_path: Path) -> None:
    files = [_write(tmp_path / "in" / n, n.encode()) for n in ("a.jpg", "b.png", "c.mp4")]
    bundle = import_files(project, files, source="folder", now=NOW)
    loaded = CaptureBundle.load(bundle.root)
    assert loaded.to_dict() == bundle.to_dict()
    assert [p.name for p in loaded.images] == ["a.jpg", "b.png"]
    assert [p.name for p in loaded.videos] == ["c.mp4"]


def test_excluded_files_are_kept_but_not_used(project: Project, tmp_path: Path) -> None:
    files = [_write(tmp_path / "in" / n, n.encode()) for n in ("a.jpg", "b.jpg", "c.mp4")]
    bundle = import_files(project, files, source="folder", now=NOW)
    bundle.set_excluded(["b.jpg", "c.mp4"])
    loaded = CaptureBundle.load(bundle.root)
    assert [p.name for p in loaded.images] == ["a.jpg"]
    assert loaded.videos == []
    assert [f.name for f in loaded.excluded] == ["b.jpg", "c.mp4"]
    assert (loaded.root / "b.jpg").read_bytes() == b"b.jpg"  # still there, untouched
    assert loaded.verify() == []
    loaded.set_excluded(["b.jpg"], excluded=False)
    assert [p.name for p in CaptureBundle.load(bundle.root).images] == ["a.jpg", "b.jpg"]
    with pytest.raises(CaptureError, match="no file 'z.jpg'"):
        loaded.set_excluded(["z.jpg"])


def test_capture_json_without_excluded_field_uses_every_file(
    project: Project, tmp_path: Path
) -> None:
    bundle = import_files(project, [_write(tmp_path / "a.jpg", b"a")], source="folder", now=NOW)
    data = json.loads((bundle.root / "capture.json").read_text())
    del data["files"][0]["excluded"]
    (bundle.root / "capture.json").write_text(json.dumps(data))
    assert [p.name for p in CaptureBundle.load(bundle.root).images] == ["a.jpg"]
    data["files"][0]["excluded"] = "yes"
    (bundle.root / "capture.json").write_text(json.dumps(data))
    with pytest.raises(CaptureError, match="excluded"):
        CaptureBundle.load(bundle.root)


def test_name_collisions_get_suffix(project: Project, tmp_path: Path) -> None:
    a = _write(tmp_path / "day1" / "IMG_0001.jpg", b"one")
    b = _write(tmp_path / "day2" / "img_0001.JPG", b"two")
    bundle = import_files(project, [a, b], source="folder", now=NOW)
    assert [f.name for f in bundle.files] == ["IMG_0001.jpg", "img_0001-2.JPG"]
    assert bundle.files[1].original_name == "img_0001.JPG"
    assert (bundle.root / "img_0001-2.JPG").read_bytes() == b"two"


def test_same_second_imports_get_distinct_ids(project: Project, tmp_path: Path) -> None:
    src = _write(tmp_path / "a.jpg", b"x")
    first = import_files(project, [src], source="folder", now=NOW)
    second = import_files(project, [src], source="folder", now=NOW)
    assert (first.id, second.id) == ("20261005-203200", "20261005-203200-2")
    assert [b.id for b in list_bundles(project)] == [first.id, second.id]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([], "nothing to import"),
        (["notes.txt"], "not a supported"),
        (["missing.jpg"], "not a file"),
    ],
)
def test_import_rejects_bad_input(
    project: Project, tmp_path: Path, names: list[str], message: str
) -> None:
    _write(tmp_path / "notes.txt", b"x")
    with pytest.raises(CaptureError, match=message):
        import_files(project, [tmp_path / n for n in names], source="folder")


def test_import_rejects_duplicates(project: Project, tmp_path: Path) -> None:
    src = _write(tmp_path / "a.jpg", b"x")
    with pytest.raises(CaptureError, match="listed twice"):
        import_files(project, [src, tmp_path / "." / "a.jpg"], source="folder")


def test_failed_import_leaves_nothing(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [_write(tmp_path / n, b"x") for n in ("a.jpg", "b.jpg")]
    real_copy = capture_mod._copy_and_hash
    calls = 0

    def flaky_copy(src: Path, dst: Path) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return real_copy(src, dst)

    monkeypatch.setattr(capture_mod, "_copy_and_hash", flaky_copy)
    with pytest.raises(OSError, match="disk full"):
        import_files(project, files, source="folder", now=NOW)
    assert list(project.captures_dir.iterdir()) == []


def test_list_bundles_ignores_incomplete_imports(project: Project, tmp_path: Path) -> None:
    (project.captures_dir / ".importing-20260101-000000").mkdir()
    (project.captures_dir / "no-manifest").mkdir()
    assert list_bundles(project) == []


def test_import_folder(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "photos"
    for name in ("b.jpg", "a.jpg", "readme.txt", ".DS_Store", "._a.jpg"):
        _write(folder / name, name.encode())
    _write(folder / "sub" / "c.jpg", b"c")
    bundle, skipped = import_folder(project, folder, now=NOW)
    assert [f.name for f in bundle.files] == ["a.jpg", "b.jpg"]
    assert skipped == [folder / "readme.txt"]
    assert bundle.source_info == {"folder": str(folder.absolute())}


def test_import_folder_without_images(project: Project, tmp_path: Path) -> None:
    _write(tmp_path / "photos" / "readme.txt", b"x")
    with pytest.raises(CaptureError, match="no supported"):
        import_folder(project, tmp_path / "photos")


def test_verify_detects_changes(project: Project, tmp_path: Path) -> None:
    files = [_write(tmp_path / n, n.encode()) for n in ("a.jpg", "b.jpg", "c.jpg")]
    bundle = import_files(project, files, source="folder", now=NOW)
    assert bundle.verify() == []
    (bundle.root / "a.jpg").unlink()
    (bundle.root / "b.jpg").write_bytes(b"longer than before")
    (bundle.root / "c.jpg").write_bytes(b"C.jpg")
    assert bundle.verify() == ["a.jpg: missing", "b.jpg: size changed", "c.jpg: contents changed"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=99),
        lambda d: d.pop("id"),
        lambda d: d.update(files="nope"),
        lambda d: d["files"][0].update(name="../escape.jpg"),
        lambda d: d["files"][0].update(kind="audio"),
        lambda d: d["files"][0].pop("sha256"),
    ],
)
def test_load_rejects_invalid_capture_json(
    project: Project, tmp_path: Path, mutate: object
) -> None:
    bundle = import_files(project, [_write(tmp_path / "a.jpg", b"x")], source="folder")
    path = bundle.root / "capture.json"
    data = json.loads(path.read_text())
    assert callable(mutate)
    mutate(data)
    path.write_text(json.dumps(data))
    with pytest.raises(CaptureError):
        CaptureBundle.load(bundle.root)


def test_import_masks(project: Project, tmp_path: Path) -> None:
    from ez2digitize.core.capture import import_masks

    a = _write(tmp_path / "day1" / "IMG_1.jpg", b"1")
    b = _write(tmp_path / "day2" / "IMG_1.jpg", b"2")  # renamed IMG_1-2.jpg in the bundle
    c = _write(tmp_path / "day2" / "IMG_3.jpg", b"3")
    bundle = import_files(project, [a, b, c], source="folder", now=NOW)
    masks = tmp_path / "masks"
    _write(masks / "IMG_1.jpg.png", b"m1")
    _write(masks / "IMG_3.png", b"m3")  # extension replaced: also accepted
    assert import_masks(project, bundle, masks) == ["IMG_1.jpg", "IMG_1-2.jpg", "IMG_3.jpg"]
    target = project.masks_dir / bundle.id
    assert sorted(p.name for p in target.iterdir()) == [
        "IMG_1-2.jpg.png",
        "IMG_1.jpg.png",
        "IMG_3.jpg.png",
    ]
