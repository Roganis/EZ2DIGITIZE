# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import random
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import ExifTags, Image, ImageFilter

from ez2digitize.core import photos
from ez2digitize.core.capture import CaptureBundle, import_files, list_bundles
from ez2digitize.core.photos import Finding, PhotoInfo, check_capture, check_project
from ez2digitize.core.project import Project

CANON = PhotoInfo(
    width=4272,
    height=2848,
    make="Canon",
    model="Canon EOS REBEL T3",
    lens="EF-S18-55mm",
    focal_mm=39.0,
    sharpness=40.0,
)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def write_photo(
    path: Path,
    *,
    size: tuple[int, int] = (640, 480),
    blur: float = 0,
    seed: int = 0,
    exif: Mapping[int, object] | None = None,
    details: Mapping[int, object] | None = None,
) -> Path:
    """A JPEG of random blocks (texture to measure sharpness on), with EXIF."""
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    for x in range(0, size[0], 8):
        for y in range(0, size[1], 8):
            image.paste(tuple(rng.randrange(256) for _ in range(3)), (x, y, x + 8, y + 8))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    tags = image.getexif()
    for tag, value in (exif or {}).items():
        tags[tag] = value
    if details:
        sub = tags.get_ifd(ExifTags.IFD.Exif)
        sub.update(details)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=95, exif=tags)
    return path


CAMERA_EXIF: dict[int, object] = {
    ExifTags.Base.Make: "Canon",
    ExifTags.Base.Model: "Canon EOS REBEL T3 \x00",
}
LENS_EXIF: dict[int, object] = {
    ExifTags.Base.FocalLength: 39.0,
    ExifTags.Base.LensModel: "EF-S18-55mm     ",
    ExifTags.Base.FocalLengthIn35mmFilm: 62,
}


# --- inspection ------------------------------------------------------------------


def test_inspect_reads_size_exif_and_sharpness(tmp_path: Path) -> None:
    info = photos.inspect_photo(
        write_photo(tmp_path / "a.jpg", exif=CAMERA_EXIF, details=LENS_EXIF)
    )
    assert info.error is None
    assert info.size == (640, 480)
    assert (info.make, info.model, info.lens) == ("Canon", "Canon EOS REBEL T3", "EF-S18-55mm")
    assert (info.focal_mm, info.focal_35mm) == (39.0, 62.0)
    assert info.sharpness is not None and info.sharpness > 0


def test_blur_lowers_sharpness(tmp_path: Path) -> None:
    sharp = photos.inspect_photo(write_photo(tmp_path / "sharp.jpg"))
    blurred = photos.inspect_photo(write_photo(tmp_path / "blurred.jpg", blur=3))
    assert sharp.sharpness and blurred.sharpness
    assert blurred.sharpness < photos.BLUR_RATIO * sharp.sharpness


def test_inspect_without_exif(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    Image.new("RGB", (300, 200), "gray").save(path)
    info = photos.inspect_photo(path)
    assert info.size == (300, 200)
    assert info.make is None and info.focal_mm is None and info.error is None


def test_unreadable_file_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg")
    info = photos.inspect_photo(path)
    assert info.error and info.size == (0, 0)


def test_info_round_trips_and_rejects_old_versions() -> None:
    assert PhotoInfo.from_dict(CANON.to_dict()) == CANON
    assert PhotoInfo.from_dict({**CANON.to_dict(), "version": 0}) is None
    assert PhotoInfo.from_dict({"version": photos.INSPECT_VERSION, "bogus": 1}) is None
    assert PhotoInfo.from_dict(None) is None


def test_inspect_bundle_stores_results_once(project: Project, tmp_path: Path) -> None:
    files = [write_photo(tmp_path / "in" / f"{i}.jpg", seed=i) for i in range(3)]
    bundle = import_files(project, files, source="folder")
    progress: list[tuple[int, int]] = []
    assert photos.needs_inspection(bundle)
    assert photos.inspect_bundle(bundle, on_progress=lambda d, t: progress.append((d, t))) == 3
    assert progress == [(1, 3), (2, 3), (3, 3)]
    saved = json.loads((bundle.root / "capture.json").read_text())
    assert saved["files"][0]["metadata"]["photo"]["width"] == 640
    reloaded = CaptureBundle.load(bundle.root)
    assert not photos.needs_inspection(reloaded)
    assert photos.inspect_bundle(reloaded) == 0


# --- checks ------------------------------------------------------------------------


def codes(findings: list[Finding]) -> list[str]:
    return [f.code for f in findings]


def same_set(count: int, **changes: object) -> dict[str, PhotoInfo]:
    return {f"{i:02}.jpg": replace(CANON, **changes) for i in range(count)}  # type: ignore[arg-type]


def test_a_good_set_has_no_findings() -> None:
    assert check_capture("c", same_set(30)) == []


def test_odd_size_photo_is_flagged_once() -> None:
    infos = same_set(30)
    infos["Preview.jpg"] = PhotoInfo(width=1207, height=737, sharpness=1800.0)
    findings = check_capture("c", infos)
    assert codes(findings) == ["odd-size"]
    assert findings[0].files == ("Preview.jpg",)
    assert "4272×2848" in findings[0].message


def test_half_and_half_sizes_are_not_odd() -> None:
    infos = same_set(4) | {f"p{i}.jpg": replace(CANON, width=3000) for i in range(4)}
    assert "odd-size" not in codes(check_capture("c", infos))


def test_unreadable() -> None:
    infos = same_set(5) | {"bad.jpg": PhotoInfo(error="cannot identify image file")}
    (finding,) = check_capture("c", infos)
    assert (finding.level, finding.code, finding.files) == ("error", "unreadable", ("bad.jpg",))


def test_low_resolution() -> None:
    findings = check_capture("c", same_set(5, width=800, height=600))
    assert codes(findings) == ["low-resolution"]


def test_no_focal_length_for_all_or_some() -> None:
    (everyone,) = check_capture("c", same_set(5, focal_mm=None))
    assert everyone.code == "no-focal-length" and everyone.files == ()
    infos = same_set(5) | {"edited.jpg": replace(CANON, focal_mm=None)}
    (some,) = check_capture("c", infos)
    assert some.code == "no-focal-length" and some.files == ("edited.jpg",)


def test_mixed_cameras_but_not_a_nudged_zoom() -> None:
    nudged = same_set(5) | {"z.jpg": replace(CANON, focal_mm=40.0)}
    assert check_capture("c", nudged) == []
    zoomed = same_set(5) | {"z.jpg": replace(CANON, focal_mm=55.0)}
    (finding,) = check_capture("c", zoomed)
    assert finding.code == "mixed-cameras"
    assert "39 mm" in finding.message and "55 mm" in finding.message
    phone = same_set(5) | {"p.jpg": replace(CANON, make="Google", model="Pixel 8", lens=None)}
    assert codes(check_capture("c", phone)) == ["mixed-cameras"]


def test_blurry_relative_to_the_set() -> None:
    infos = same_set(10) | {"shake.jpg": replace(CANON, sharpness=10.0)}
    (finding,) = check_capture("c", infos)
    assert finding.code == "blurry" and finding.files == ("shake.jpg",)
    # Too few photos to know what sharp means for this subject.
    assert check_capture("c", same_set(3) | {"shake.jpg": replace(CANON, sharpness=10.0)}) == []


def test_project_checks(project: Project, tmp_path: Path) -> None:
    first = write_photo(tmp_path / "in" / "a.jpg", exif=CAMERA_EXIF, details=LENS_EXIF)
    import_files(project, [first], source="folder")
    import_files(project, [first], source="folder")  # the same photo again
    bundles = list_bundles(project)
    # Not inspected yet: only what needs no inspection.
    assert codes(check_project(bundles)) == ["duplicate"]
    for bundle in bundles:
        photos.inspect_bundle(bundle)
    findings = check_project(list_bundles(project))
    assert codes(findings) == ["low-resolution", "low-resolution", "duplicate", "few-photos"]
    duplicate = findings[2]
    assert duplicate.capture is None and len(duplicate.files) == 2


def test_excluded_photos_are_not_checked(project: Project, tmp_path: Path) -> None:
    first = write_photo(tmp_path / "in" / "a.jpg")
    import_files(project, [first], source="folder")
    second = import_files(project, [first], source="folder")
    second.set_excluded(["a.jpg"])
    assert "duplicate" not in codes(check_project(list_bundles(project)))


def test_camera_groups(project: Project, tmp_path: Path) -> None:
    from ez2digitize.core.photos import PHOTO_KEY, camera_groups

    files = [write_photo(tmp_path / "in" / f"{i}.jpg", seed=i) for i in range(4)]
    bundle = import_files(project, files, source="folder")
    infos = [CANON, replace(CANON, focal_mm=40.0), replace(CANON, focal_mm=55.0), CANON]
    for entry, info in zip(bundle.files, infos, strict=True):
        entry.metadata[PHOTO_KEY] = info.to_dict()
    groups = camera_groups([bundle])
    by_name = {name.split("/")[1]: group for name, group in groups.items()}
    # 39 and 40 mm are one zoom setting; 55 mm another.
    assert by_name["0.jpg"] == by_name["1.jpg"] == by_name["3.jpg"] != by_name["2.jpg"]
    # One camera per capture: nothing to do.
    for entry in bundle.files:
        entry.metadata[PHOTO_KEY] = CANON.to_dict()
    assert camera_groups([bundle]) == {}


def test_inspect_finds_gps(tmp_path: Path) -> None:
    path = write_photo(tmp_path / "outdoors.jpg")
    with Image.open(path) as image:
        tags = image.getexif()
        tags.get_ifd(ExifTags.IFD.GPSInfo).update(
            {
                ExifTags.GPS.GPSLatitudeRef: "N",
                ExifTags.GPS.GPSLatitude: (48.0, 51.0, 30.0),
                ExifTags.GPS.GPSLongitudeRef: "E",
                ExifTags.GPS.GPSLongitude: (2.0, 17.0, 40.0),
            }
        )
        image.save(tmp_path / "tagged.jpg", exif=tags)
    assert photos.inspect_photo(tmp_path / "tagged.jpg").gps
    assert not photos.inspect_photo(path).gps


def test_inspect_reads_exposure(tmp_path: Path) -> None:
    exposure: dict[int, object] = {
        ExifTags.Base.ExposureTime: 1 / 2000,
        ExifTags.Base.FNumber: 1.8,
        ExifTags.Base.ISOSpeedRatings: 400,
        ExifTags.Base.ExposureBiasValue: -0.7,
        ExifTags.Base.WhiteBalance: 1,
    }
    info = photos.inspect_photo(write_photo(tmp_path / "a.jpg", details=exposure))
    assert info.exposure_s == pytest.approx(1 / 2000)  # not rounded away
    assert (info.f_number, info.iso, info.exposure_bias) == (1.8, 400.0, -0.7)
    assert info.white_balance == "manual"
    plain = photos.inspect_photo(write_photo(tmp_path / "b.jpg"))
    assert plain.exposure_s is None and plain.white_balance is None


def _exposures(*shutters: float) -> dict[str, PhotoInfo]:
    return {
        f"{n:02}.jpg": replace(CANON, exposure_s=t, f_number=8.0, iso=100.0)
        for n, t in enumerate(shutters)
    }


def test_exposure_changes_are_reported() -> None:
    steady = _exposures(*[1 / 100] * 8, 1 / 125)  # a third of a stop: fine
    assert check_capture("c", steady) == []
    drifting = _exposures(*[1 / 100] * 8, 1 / 400, 1 / 25)  # two stops either way
    (finding,) = check_capture("c", drifting)
    assert finding.code == "exposure-changes" and finding.files == ("08.jpg", "09.jpg")
    assert "up to 4.0 stops" in finding.message and "lock the exposure" in finding.message
    stops = photos.exposure_stops(drifting)
    assert stops["09.jpg"] - stops["00.jpg"] == pytest.approx(2.0)


def test_exposure_report(project: Project, tmp_path: Path) -> None:
    files = [write_photo(tmp_path / f"{n}.jpg", seed=n) for n in range(5)]
    bundle = import_files(project, files, source="folder")
    for n, entry in enumerate(bundle.files):
        info = replace(
            CANON, exposure_s=1 / 100 if n < 4 else 1 / 800, iso=100.0, white_balance="auto"
        )
        entry.metadata[photos.PHOTO_KEY] = info.to_dict()
    bundle.save()
    names = sorted(f.name for f in bundle.files)
    placed = {f"{bundle.id}/{n}" for n in names[:3]}  # 3.jpg and 4.jpg not placed
    weak = [f"{bundle.id}/{names[0]}"]
    (report,) = photos.exposure_report(list_bundles(project), placed, weak)
    assert report.photos == 5 and report.with_exposure == 5
    assert report.spread_stops == 3.0 and report.off == ("4.jpg",)
    assert report.shutter_s == (1 / 800, 1 / 100) and report.white_balance == ("auto",)
    assert report.unplaced == (1, 1)  # the dark one, and one of the four others
    assert report.weak == (0, 1)
    (bare,) = photos.exposure_report(list_bundles(project))
    assert bare.unplaced is None
