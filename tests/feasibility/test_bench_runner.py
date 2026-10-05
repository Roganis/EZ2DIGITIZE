# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from ez2d_bench.frames import pick_sharpest, sharpness
from ez2d_bench.record import RunRecord
from ez2d_bench.report import fmt_duration, fmt_mb, render
from ez2d_bench.runner import StepResult, capture, run_step


def test_run_step_records_exit_code_output_and_memory(tmp_path: Path) -> None:
    script = "import sys; print('hello'); print('world', file=sys.stderr); sys.exit(3)"
    result = run_step("demo", [sys.executable, "-c", script], tmp_path / "demo.log", echo=False)
    assert result.exit_code == 3
    assert not result.ok
    assert result.tail == ["hello", "world"] or sorted(result.tail) == ["hello", "world"]
    assert "hello" in (tmp_path / "demo.log").read_text()
    assert result.peak_rss_mb > 1
    assert result.wall_s >= 0


def test_capture_returns_output_even_on_failure() -> None:
    out = capture([sys.executable, "-c", "import sys; print('usage: x'); sys.exit(1)"])
    assert "usage: x" in out
    assert "failed to run" in capture(["/nonexistent/binary"])


def test_pick_sharpest_per_window() -> None:
    assert pick_sharpest([1, 5, 2, 9, 3, 4, 7], window=3) == [1, 3, 6]


def test_sharpness_prefers_sharp_image(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    sharp = Image.fromarray((rng.random((200, 300)) * 255).astype(np.uint8))
    sharp.save(tmp_path / "sharp.png")
    sharp.filter(ImageFilter.GaussianBlur(3)).save(tmp_path / "blurry.png")
    assert sharpness(tmp_path / "sharp.png") > 10 * sharpness(tmp_path / "blurry.png")


def test_formatting() -> None:
    assert fmt_duration(None) == "–"
    assert fmt_duration(42) == "42s"
    assert fmt_duration(125) == "2m 05s"
    assert fmt_duration(3 * 3600 + 7 * 60) == "3h 07m"
    assert fmt_mb(512) == "512 MB"
    assert fmt_mb(3072) == "3.0 GB"


def _step(name: str, wall: float, rss: float = 100.0) -> StepResult:
    return StepResult(name=name, command=[name], exit_code=0, wall_s=wall, cpu_s=wall,
                      peak_rss_mb=rss)  # fmt: skip


def test_report_renders_ok_and_failed_runs() -> None:
    ok = RunRecord(
        kind="colmap", label="sift3200", dataset="mug", machine="arch", status="ok",
        config={"max_image_size": 3200, "matcher": "exhaustive", "mapper": "incremental",
                "masks": False},
        steps=[_step("feature_extractor", 30), _step("exhaustive_matcher", 90, 2048),
               _step("mapper", 60)],
        metrics={"registered_images": 58, "total_images": 60, "models": 1, "points": 41234,
                 "mean_reprojection_error_px": 0.61},
    )  # fmt: skip
    failed = RunRecord(
        kind="openmvs", label="lvl0", dataset="mug", machine="arch", status="failed",
        reason="step 'DensifyPointCloud' exited with code -9", config={},
        steps=[StepResult("DensifyPointCloud", [], -9, 5.0, 5.0, 7000.0, tail=["Killed"])],
    )  # fmt: skip
    md = render("arch", {"os": "Arch Linux", "gpus": [{"deviceName": "RX 7900 GRE"}],
                         "tools": {"colmap": "3.12"}}, [ok, failed])  # fmt: skip
    assert "| mug | sift3200 | 3200 | exhaustive | incremental | no | 58/60 | 1 | 41,234 |" in md
    assert "3m 00s" in md  # total time
    assert "2.0 GB" in md  # peak RAM of the largest step
    assert "lvl0" in md and "exited with code -9" in md and "Killed" in md
    assert "RX 7900 GRE" in md


def test_peak_rss_is_the_commands_not_the_callers(tmp_path: Path) -> None:
    ballast = bytearray(400 * 2**20)  # make this (the parent) process large
    ballast[::4096] = b"x" * len(ballast[::4096])
    small = run_step("small", ["true"], tmp_path / "small.log", echo=False)
    big_script = "b = bytearray(200 * 2**20); b[::4096] = b'x' * len(b[::4096])"
    big = run_step("big", [sys.executable, "-c", big_script], tmp_path / "big.log", echo=False)
    assert small.peak_rss_mb < 100
    assert 200 < big.peak_rss_mb < 380
    del ballast


def test_signal_exit_is_reported_negative(tmp_path: Path) -> None:
    script = "import os, signal; os.kill(os.getpid(), signal.SIGABRT)"
    result = run_step("abort", [sys.executable, "-c", script], tmp_path / "a.log", echo=False)
    assert result.exit_code == -6


def test_missing_executable_fails_cleanly(tmp_path: Path) -> None:
    result = run_step("missing", ["/nonexistent/tool"], tmp_path / "m.log", echo=False)
    assert result.exit_code == 127
    assert "cannot run" in (tmp_path / "m.log").read_text()


def test_tool_versions_are_merged_across_runs() -> None:
    from ez2d_bench.report import merged_tool_versions

    a = RunRecord("colmap", "a", "d", "m", {}, tool_versions={"colmap": "3.9", "openmvs": None})
    b = RunRecord("openmvs", "b", "d", "m", {}, tool_versions={"colmap": "3.9", "openmvs": "2.3"})
    merged = merged_tool_versions({"tools": {"brush": None, "colmap": "3.12"}}, [a, b])
    assert merged == {"colmap": "3.9 / 3.12", "openmvs": "2.3", "brush": None}


def test_cli_makes_paths_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bench

    seen: dict[str, Path] = {}

    def fake_synth(args: object) -> int:
        seen["out"] = args.out  # type: ignore[attr-defined]
        return 0

    monkeypatch.chdir(tmp_path)
    parser_args = ["synth", "rel/dir"]
    monkeypatch.setattr(bench, "cmd_synth", fake_synth)
    assert bench.main(parser_args) == 0
    assert seen["out"] == tmp_path / "rel" / "dir"
