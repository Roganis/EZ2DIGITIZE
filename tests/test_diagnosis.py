# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from pathlib import Path

import pytest

from ez2digitize.core.stage import Backend, StageManifest
from ez2digitize.diagnosis import explain
from ez2digitize.pipeline import StageFailed

# How a process ends when it runs out of memory: SIGKILL from the kernel's
# OOM killer, or Windows' STATUS_NO_MEMORY.
OUT_OF_MEMORY = 0xC0000017 if sys.platform == "win32" else -9


@pytest.mark.parametrize(
    ("stage", "code", "log", "title"),
    [
        ("features", OUT_OF_MEMORY, [], "The step ran out of memory"),
        ("features", 0xC000012D, [], "out of memory"),
        ("densify", 1, ["terminate called after throwing 'std::bad_alloc'"], "out of memory"),
        ("densify", 1, ["error: writing: No space left on device"], "The disk is full"),
        ("mapping", 1, ["E1005 12:00:00.1 1 m.cc:9] Failed to create any sparse model"], "placed"),
        ("mapping", 1, ["E1005 12:00:00.1 1 g.cc:9] Global mapping failed"], "placed"),
        ("matching", 1, ["Cannot continue without image pairs"], "placed"),
        ("features", 1, ["W1005 x.cc:1] Could not read image a/b.jpg"], "could not be read"),
        ("densify", 1, ["error: no images see 3 or more points from 9 selected"], "dense"),
        ("mesh", 1, ["error: empty initial mesh"], "No mesh"),
        ("texture", -11, [], "crashed"),
        ("mesh", 139, [], "crashed"),
        ("features", -4, [], "newer processor"),
    ],
)
def test_known_failures(stage: str, code: int, log: list[str], title: str) -> None:
    found = explain(stage, code, log)
    assert found is not None and title.lower() in found.title.lower()


def test_unknown_failures_are_not_guessed() -> None:
    assert explain("mapping", 1, ["something went wrong"]) is None
    # A pattern only counts in the step it belongs to.
    assert explain("texture", 1, ["Failed to create any sparse model"]) is None


def test_memory_wins_over_the_crash_it_causes() -> None:
    found = explain("densify", -6, ["std::bad_alloc", "Aborted"])
    assert found is not None and "memory" in found.title


def _manifest(stage: str, code: int) -> StageManifest:
    return StageManifest(
        stage=stage, run_id="r", status="failed", cache_key="k", backend=Backend("x", "1"),
        command=[], parameters={}, inputs={}, started="", finished="", wall_s=0, cpu_s=0,
        peak_rss_mb=None, exit_code=code, host={},
    )  # fmt: skip


def test_stage_failed_message() -> None:
    explained = StageFailed(_manifest("features", OUT_OF_MEMORY), Path("log.txt"), [])
    assert str(explained).startswith("The step ran out of memory (stage features failed")
    assert "lower detail level" in str(explained)
    plain = StageFailed(_manifest("features", 3), Path("log.txt"), ["?"])
    assert str(plain) == "stage features failed (exit code 3)" and plain.explanation is None
