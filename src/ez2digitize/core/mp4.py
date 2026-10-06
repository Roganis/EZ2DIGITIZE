# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Just enough of the MP4 (ISO base media) format to read a metadata track's samples.

Cameras store motion data as a timed track next to the video: GoPro's
GPMF (sample entry `gpmd`) and Google's CAMM (`camm`). Reading one needs
the track's sample table: where each sample lies in the file and when it
starts. This walks the boxes (`moov/trak/mdia/minf/stbl`) without reading
the media data, so large videos cost only the size of their index.

Edit lists are not applied: times are from the start of each track's own
timeline, which for camera files starts with the video's.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"})
# A metadata index larger than this is not an MP4 we can read sensibly.
MAX_MOOV_BYTES = 256 * 1024 * 1024


class Mp4Error(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    offset: int  # in the file
    size: int
    time_s: float  # start, from the track's start
    duration_s: float


@dataclass
class Track:
    track_id: int = 0
    handler: str = ""  # "vide", "soun", "meta", ...
    format: str = ""  # the first sample entry's: "avc1", "gpmd", "camm", ...
    timescale: int = 0
    samples: list[Sample] = field(default_factory=list)


def read_tracks(path: Path) -> list[Track]:
    """Every track of the file, with its sample table.

    None of them for a file that isn't MP4 or MOV (Matroska, AVI...), which
    has no moov box. Raises Mp4Error for a corrupt one.
    """
    with path.open("rb") as f:
        try:
            moov = _find_top_level(f, b"moov")
        except Mp4Error:
            return []  # not boxes at all: another format
        if moov is None:
            return []
        tracks = []
        for kind, payload in _boxes(moov):
            if kind == b"trak":
                tracks.append(_track(payload))
        return tracks


def read_samples(path: Path, track: Track) -> Iterator[tuple[Sample, bytes]]:
    """Each sample of `track` with its bytes, in order."""
    with path.open("rb") as f:
        for sample in track.samples:
            f.seek(sample.offset)
            data = f.read(sample.size)
            if len(data) != sample.size:
                raise Mp4Error(f"{path.name}: a sample lies past the end of the file")
            yield sample, data


def _find_top_level(f: BinaryIO, wanted: bytes) -> bytes | None:
    f.seek(0, 2)
    end = f.tell()
    position = 0
    while position + 8 <= end:
        f.seek(position)
        header = f.read(8)
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            (size,) = struct.unpack(">Q", f.read(8))
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size:
            raise Mp4Error("corrupt box size")
        if kind == wanted:
            if size - header_size > MAX_MOOV_BYTES:
                raise Mp4Error("the moov box is too large")
            data = f.read(size - header_size)
            if len(data) != size - header_size:
                raise Mp4Error("the file ends inside the moov box")
            return data
        position += size
    return None


def _boxes(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """(type, payload) of the boxes packed in `data`."""
    position = 0
    while position + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, position)
        header_size = 8
        if size == 1:
            if position + 16 > len(data):
                raise Mp4Error("corrupt box header")
            (size,) = struct.unpack_from(">Q", data, position + 8)
            header_size = 16
        elif size == 0:
            size = len(data) - position
        if size < header_size or position + size > len(data):
            raise Mp4Error("corrupt box size")
        yield kind, data[position + header_size : position + size]
        position += size


def _walk(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """Every box under `data`, descending into the containers."""
    for kind, payload in _boxes(data):
        yield kind, payload
        if kind in CONTAINERS:
            yield from _walk(payload)


def _track(trak: bytes) -> Track:
    track = Track()
    deltas: list[tuple[int, int]] = []
    chunks_runs: list[tuple[int, int]] = []
    sizes: list[int] = []
    offsets: list[int] = []
    try:
        for kind, box in _walk(trak):
            version = box[0] if box else 0
            if kind == b"tkhd":
                (track.track_id,) = struct.unpack_from(">I", box, 20 if version == 1 else 12)
            elif kind == b"mdhd":
                (track.timescale,) = struct.unpack_from(">I", box, 20 if version == 1 else 12)
            elif kind == b"hdlr":
                track.handler = box[8:12].decode("latin-1")
            elif kind == b"stsd":
                (count,) = struct.unpack_from(">I", box, 4)
                if count:
                    track.format = box[12:16].decode("latin-1")
            elif kind == b"stts":
                (count,) = struct.unpack_from(">I", box, 4)
                deltas = [struct.unpack_from(">II", box, 8 + 8 * i) for i in range(count)]
            elif kind == b"stsc":
                (count,) = struct.unpack_from(">I", box, 4)
                chunks_runs = [struct.unpack_from(">II", box, 8 + 12 * i) for i in range(count)]
            elif kind == b"stsz":
                uniform, count = struct.unpack_from(">II", box, 4)
                if uniform:
                    sizes = [uniform] * count
                else:
                    sizes = list(struct.unpack_from(f">{count}I", box, 12))
            elif kind == b"stco":
                (count,) = struct.unpack_from(">I", box, 4)
                offsets = list(struct.unpack_from(f">{count}I", box, 8))
            elif kind == b"co64":
                (count,) = struct.unpack_from(">I", box, 4)
                offsets = list(struct.unpack_from(f">{count}Q", box, 8))
    except struct.error as exc:
        raise Mp4Error(f"corrupt track header: {exc}") from exc
    if track.timescale <= 0:
        return track
    track.samples = _samples(sizes, offsets, chunks_runs, deltas, track.timescale)
    return track


def _samples(
    sizes: list[int],
    offsets: list[int],
    chunk_runs: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    timescale: int,
) -> list[Sample]:
    """The sample table spelled out: chunks hold runs of consecutive samples."""
    per_chunk = []
    for index, (first, count) in enumerate(chunk_runs):
        last = chunk_runs[index + 1][0] if index + 1 < len(chunk_runs) else len(offsets) + 1
        per_chunk += [count] * max(0, last - first)
    durations = [delta for count, delta in deltas for _ in range(count)]
    samples: list[Sample] = []
    ticks = 0
    for chunk, count in enumerate(per_chunk):
        position = offsets[chunk]
        for _ in range(count):
            n = len(samples)
            if n >= len(sizes):
                return samples
            duration = durations[n] if n < len(durations) else 0
            samples.append(Sample(position, sizes[n], ticks / timescale, duration / timescale))
            position += sizes[n]
            ticks += duration
    return samples
