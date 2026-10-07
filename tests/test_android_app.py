# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Android capture app (android/) and the desktop agree on the motion log.

The app's unit test (MotionLogTest) writes its log for fixed readings and
checks it against android/app/src/test/resources/sample.motion.json; here
the desktop reads that same file.
"""

import json
from pathlib import Path

import pytest

from ez2digitize.motion_log import parse_log, read_log

SAMPLE = Path(__file__).parents[1] / "android/app/src/test/resources/sample.motion.json"


def test_the_apps_log_reads() -> None:
    log = read_log(SAMPLE)
    assert log.frames == {"IMG_0001.jpg": pytest.approx(12345.682901)}
    track = log.track
    assert track.device == 'Google Pixel "8"' and track.metric
    assert track.gravity == "gravity sensor"
    # The phone upright in portrait: gravity along the stored image's +x.
    assert track.down_at(12345.682901) == pytest.approx((1.0, 0.0, 0.0))
    assert track.turn_at(12345.682901) == pytest.approx(28.65, abs=0.01)
    pose = track.pose_at(12345.682901)
    assert pose is not None
    rotation, centre = pose
    assert rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert centre == (0.25, -1.5, 0.5)  # ARCore's (0.25, 1.5, -0.5), y and z flipped


def test_empty_lists_are_valid() -> None:
    """A log whose phone gave no sensor readings, as the app writes it."""
    text = (
        '{\n"format": "ez2digitize-motion", "version": 1,\n'
        '"device": {"make": "x", "model": "y"},\n"metric": true,\n'
        '"gyroscope": [\n],\n"accelerometer": [\n],\n"gravity": [\n],\n"poses": [\n],\n'
        '"frames": [\n  {"file": "IMG_0001.jpg", "t": 1.5}\n]\n}\n'
    )
    log = parse_log(json.loads(text))
    assert log.frames == {"IMG_0001.jpg": 1.5} and not log.track.down
