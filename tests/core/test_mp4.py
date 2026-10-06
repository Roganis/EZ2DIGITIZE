# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from mp4_files import TrackSpec, write_mp4

from ez2digitize.core.mp4 import Mp4Error, read_samples, read_tracks


@pytest.mark.parametrize("co64", [False, True])
def test_sample_table(tmp_path: Path, co64: bool) -> None:
    video = TrackSpec("vide", "avc1", 30000, [(b"V" * 10, 1001)] * 3, chunk_size=3)
    meta = TrackSpec("meta", "gpmd", 1000, [(b"A" * 5, 1001), (b"BB", 1001), (b"CCC", 500)])
    path = tmp_path / "GX010001.MP4"
    write_mp4(path, [video, meta], co64=co64)
    tracks = read_tracks(path)
    assert [(t.track_id, t.handler, t.format) for t in tracks] == [
        (1, "vide", "avc1"),
        (2, "meta", "gpmd"),
    ]
    samples = list(read_samples(path, tracks[1]))
    assert [data for _, data in samples] == [b"A" * 5, b"BB", b"CCC"]
    assert [s.time_s for s, _ in samples] == pytest.approx([0, 1.001, 2.002])
    assert samples[2][0].duration_s == pytest.approx(0.5)


def test_other_formats_have_no_tracks(tmp_path: Path) -> None:
    mkv = tmp_path / "clip.mkv"
    mkv.write_bytes(b"\x1aE\xdf\xa3 pretend this is Matroska")
    assert read_tracks(mkv) == []


def test_corrupt_index(tmp_path: Path) -> None:
    path = tmp_path / "bad.mp4"
    write_mp4(path, [TrackSpec("meta", "gpmd", 1000, [(b"A" * 8, 1000)])])
    data = bytearray(path.read_bytes())
    stsz = data.index(b"stsz")
    data[stsz + 12 : stsz + 16] = (10**6).to_bytes(4, "big")  # claims a million samples
    path.write_bytes(bytes(data))
    with pytest.raises(Mp4Error):
        read_tracks(path)
