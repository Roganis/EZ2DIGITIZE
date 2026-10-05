# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

from ez2d_bench.plan import PlanError, load

PLAN_DIR = Path(__file__).resolve().parents[2] / "tools" / "feasibility" / "plans"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mug.toml"
    path.write_text(body)
    return path


def test_minimal_plan_defaults(tmp_path: Path) -> None:
    plan = load(_write(tmp_path, '[dataset]\nimages = "photos"\n[[colmap]]\nlabel = "a"\n'))
    assert plan.dataset.name == "mug"
    assert plan.dataset.images == tmp_path / "photos"
    assert plan.colmap[0].matcher == "exhaustive"
    assert plan.colmap[0].max_image_size == 3200


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[dataset]\nimages = "x"\n[[colmap]]\nlabel = "a"\nmax_size = 1\n', "unknown keys"),
        ('[dataset]\nimages = "x"\n[[openmvs]]\nlabel = "o"\nsource = "nope"\n', "nope"),
        ('[dataset]\nimages = "x"\n[[colmap]]\nlabel = "a"\nmasks = true\n', "no 'masks'"),
        ('[dataset]\nimages = "x"\n[[colmap]]\nlabel = "a"\nmatcher = "vocab"\n', "matcher"),
        (
            '[dataset]\nimages = "x"\n[[colmap]]\nlabel = "a"\n[[colmap]]\nlabel = "a"\n',
            "duplicate",
        ),
        ('[dataset]\nimages = "x"\n[colmapp]\n', "unknown sections"),
    ],
)
def test_plan_errors(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        load(_write(tmp_path, body))


@pytest.mark.parametrize("plan_file", sorted(PLAN_DIR.glob("*.toml")), ids=lambda p: p.name)
def test_shipped_plans_are_valid(plan_file: Path) -> None:
    plan = load(plan_file)
    assert plan.colmap
