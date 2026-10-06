# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small MP4 files with metadata tracks, for the motion tests: GPMF and CAMM samples."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def full_box(kind: bytes, payload: bytes, version: int = 0) -> bytes:
    return box(kind, struct.pack(">I", version << 24) + payload)


@dataclass
class TrackSpec:
    handler: str  # "meta", "vide"...
    format: str  # "gpmd", "camm"...
    timescale: int
    samples: Sequence[tuple[bytes, int]]  # (data, duration in timescale units)
    chunk_size: int = 2  # samples per chunk


def write_mp4(path: Path, tracks: Sequence[TrackSpec], *, co64: bool = False) -> None:
    """ftyp, mdat with every sample, then moov (as cameras write it)."""
    header = box(b"ftyp", b"isom\0\0\0\0isom")
    media = b"".join(data for track in tracks for data, _ in track.samples)
    mdat_start = len(header) + 8
    offset = mdat_start
    traks = []
    for track_id, track in enumerate(tracks, start=1):
        sizes = [len(data) for data, _ in track.samples]
        chunk_offsets = []
        position = offset
        for start in range(0, len(sizes), track.chunk_size):
            chunk_offsets.append(position)
            position += sum(sizes[start : start + track.chunk_size])
        offset = position
        durations = [d for _, d in track.samples]
        stts = full_box(
            b"stts",
            struct.pack(">I", len(durations))
            + b"".join(struct.pack(">II", 1, d) for d in durations),
        )
        stsc = full_box(b"stsc", struct.pack(">IIII", 1, 1, track.chunk_size, 1))
        stsz = full_box(b"stsz", struct.pack(f">II{len(sizes)}I", 0, len(sizes), *sizes))
        if co64:
            stco = full_box(
                b"co64", struct.pack(f">I{len(chunk_offsets)}Q", len(chunk_offsets), *chunk_offsets)
            )
        else:
            stco = full_box(
                b"stco", struct.pack(f">I{len(chunk_offsets)}I", len(chunk_offsets), *chunk_offsets)
            )
        entry = box(track.format.encode(), b"\0" * 8)
        stsd = full_box(b"stsd", struct.pack(">I", 1) + entry)
        stbl = box(b"stbl", stsd + stts + stsc + stsz + stco)
        minf = box(b"minf", stbl)
        hdlr = full_box(b"hdlr", b"\0" * 4 + track.handler.encode() + b"\0" * 13)
        mdhd = full_box(b"mdhd", struct.pack(">IIII", 0, 0, track.timescale, sum(durations)))
        mdia = box(b"mdia", mdhd + hdlr + minf)
        tkhd = full_box(b"tkhd", struct.pack(">IIII", 0, 0, track_id, 0) + b"\0" * 64)
        traks.append(box(b"trak", tkhd + mdia))
    moov = box(b"moov", b"".join(traks))
    path.write_bytes(header + box(b"mdat", media) + moov)


# --- GPMF ---


def klv(key: str, kind: str, size: int, values: Sequence[float | int] | bytes) -> bytes:
    """One GPMF record; numbers packed big-endian by `kind`, `size` bytes a sample."""
    if isinstance(values, bytes):
        data = values
    else:
        code = {"s": "h", "S": "H", "l": "i", "f": "f", "L": "I"}[kind]
        data = struct.pack(f">{len(values)}{code}", *values)
    repeat = len(data) // size
    padding = b"\0" * (-len(data) % 4)
    return struct.pack(">4scBH", key.encode(), kind.encode(), size, repeat) + data + padding


def nested(key: str, *children: bytes) -> bytes:
    data = b"".join(children)
    return struct.pack(">4scBH", key.encode(), b"\0", 4, len(data) // 4) + data


def gpmf_payload(device: str, *streams: bytes) -> bytes:
    name = device.encode()
    return nested("DEVC", klv("DVNM", "c", 1, name + b"\0" * (-len(name) % 4)), *streams)


# --- CAMM ---


def camm(kind: int, *values: float) -> bytes:
    return struct.pack(f"<HH{len(values)}f", 0, kind, *values)
