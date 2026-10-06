# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageOps

from ez2digitize import mask_worker


class _Input:
    name = "input_image"


class FakeSession:
    """Answers 'object' for the left third of the (upright) picture."""

    def __init__(self) -> None:
        self.batches: list[Any] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs: None, feed: dict[str, Any]) -> list[Any]:
        batch = feed["input_image"]
        self.batches.append(batch)
        size = batch.shape[-1]
        out = np.zeros((1, 1, size, size), dtype=np.float32)
        out[..., : size // 3] = 1.0
        return [out]


def _photo(path: Path, size: tuple[int, int], orientation: int) -> None:
    exif = Image.Exif()
    exif[mask_worker.ORIENTATION_TAG] = orientation
    Image.new("RGB", size, (200, 100, 50)).save(path, exif=exif)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_undo_orientation_inverts_exif_transpose(orientation: int) -> None:
    raw = Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3, 4) * 20)
    exif = raw.getexif()
    exif[mask_worker.ORIENTATION_TAG] = orientation
    raw.info["exif"] = exif.tobytes()
    upright = ImageOps.exif_transpose(raw)
    undo = mask_worker.UNDO_ORIENTATION.get(orientation)
    back = upright.transpose(undo) if undo is not None else upright
    assert np.array_equal(np.asarray(back), np.asarray(raw))


def test_mask_is_in_the_raw_layout(tmp_path: Path) -> None:
    # Stored landscape 60x40, shown upright as portrait 40x60 (orientation 6).
    photo = tmp_path / "phone.jpg"
    _photo(photo, (60, 40), 6)
    session = FakeSession()
    mask, stats = mask_worker.make_mask(
        session,
        photo,
        size=32,
        threshold=0.5,
        grow=0.0,
    )
    assert mask.mode == "L" and mask.size == (60, 40)
    assert abs(stats["coverage"] - 10 / 32) < 1e-6
    assert stats["uncertain"] == 0.0
    pixels = np.asarray(mask)
    # Orientation 6 shows the raw picture turned 90 degrees clockwise, so
    # the upright picture's left third is the raw picture's bottom third.
    assert pixels[-12:].min() == 255 and pixels[:26].max() == 0
    # The model saw the photo upright, normalised to [-0.5, 0.5].
    batch = session.batches[0]
    assert batch.shape == (1, 3, 32, 32) and batch.dtype == np.float32
    assert batch.min() >= -0.5 and batch.max() <= 0.5


def test_grow_max() -> None:
    values = np.zeros((7, 7), dtype=np.float32)
    values[3, 3] = 1.0
    grown = mask_worker.grow_max(values, 1)
    assert grown.shape == (7, 7)
    assert grown[2:5, 2:5].min() == 1.0
    assert grown.sum() == 9.0


def test_main_writes_masks_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: FakeSession())
    _photo(tmp_path / "a.jpg", (30, 20), 1)
    out = tmp_path / "out" / "a.jpg.png"
    job = {"key": "a.jpg", "image": str(tmp_path / "a.jpg"), "mask": str(out)}
    jobs = {"jobs": [job]}
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    model = tmp_path / "m.onnx"
    argv = ["--model", str(model), "--jobs", str(tmp_path / "jobs.json"), "--size", "16"]
    assert mask_worker.main(argv) == 0
    with Image.open(tmp_path / "out" / "a.jpg.png") as mask:
        assert mask.size == (30, 20)
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report) == {"a.jpg"} and report["a.jpg"]["coverage"] > 0
    assert "mask 1/1 a.jpg" in capsys.readouterr().out
