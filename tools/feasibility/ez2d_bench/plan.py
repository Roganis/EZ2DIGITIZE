# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark plans: one TOML file per dataset listing the variants to run.

See tools/feasibility/plans/example-object.toml for the format.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Matcher = Literal["exhaustive", "sequential"]
Mapper = Literal["incremental", "global"]


class PlanError(ValueError):
    pass


@dataclass
class Dataset:
    name: str
    images: Path
    masks: Path | None = None
    notes: str = ""


@dataclass
class ColmapRun:
    label: str
    max_image_size: int = 3200
    matcher: Matcher = "exhaustive"
    mapper: Mapper = "incremental"
    masks: bool = False
    camera_model: str | None = None
    single_camera: bool = True
    sequential_overlap: int = 10
    undistort_max_image_size: int | None = None
    threads: int | None = None
    # Extra arguments per COLMAP command, e.g. {feature_extractor = ["--X", "1"]}
    extra: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class OpenMVSRun:
    label: str
    source: str  # label of a ColmapRun
    resolution_level: int = 1
    max_resolution: int | None = None
    masks: bool = False
    refine: bool = False
    refine_resolution_level: int = 2
    texture: bool = True
    # OBJ export crashed in OpenMVS 2.3.0 (Mesh::SaveOBJ); PLY works everywhere.
    texture_export: Literal["ply", "obj"] = "ply"
    threads: int | None = None
    extra: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class BrushRun:
    label: str
    source: str  # label of a ColmapRun
    args: list[str] = field(default_factory=list)


@dataclass
class Plan:
    path: Path
    dataset: Dataset
    colmap: list[ColmapRun] = field(default_factory=list)
    openmvs: list[OpenMVSRun] = field(default_factory=list)
    brush: list[BrushRun] = field(default_factory=list)

    def colmap_run(self, label: str) -> ColmapRun:
        for run in self.colmap:
            if run.label == label:
                return run
        raise PlanError(f"no [[colmap]] run with label {label!r}")

    def labels(self) -> list[str]:
        runs: list[ColmapRun | OpenMVSRun | BrushRun] = [*self.colmap, *self.openmvs, *self.brush]
        return [r.label for r in runs]


def _build[T](cls: type[T], data: dict[str, Any], where: str) -> T:
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise PlanError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(names)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise PlanError(f"{where}: {exc}") from exc


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load(path: Path) -> Plan:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    base = path.parent

    unknown = set(data) - {"dataset", "colmap", "openmvs", "brush"}
    if unknown:
        raise PlanError(f"{path}: unknown sections {sorted(unknown)}")
    if "dataset" not in data:
        raise PlanError(f"{path}: missing [dataset] section")

    ds = dict(data["dataset"])
    ds.setdefault("name", path.stem)
    if "images" not in ds:
        raise PlanError(f"{path}: [dataset] needs 'images'")
    ds["images"] = _resolve(base, ds["images"])
    if ds.get("masks"):
        ds["masks"] = _resolve(base, ds["masks"])
    dataset = _build(Dataset, ds, "[dataset]")

    plan = Plan(
        path=path,
        dataset=dataset,
        colmap=[_build(ColmapRun, d, "[[colmap]]") for d in data.get("colmap", [])],
        openmvs=[_build(OpenMVSRun, d, "[[openmvs]]") for d in data.get("openmvs", [])],
        brush=[_build(BrushRun, d, "[[brush]]") for d in data.get("brush", [])],
    )
    _validate(plan)
    return plan


def _validate(plan: Plan) -> None:
    labels = plan.labels()
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise PlanError(f"duplicate labels: {sorted(duplicates)}")
    for c in plan.colmap:
        if c.matcher not in ("exhaustive", "sequential"):
            raise PlanError(f"{c.label}: matcher must be 'exhaustive' or 'sequential'")
        if c.mapper not in ("incremental", "global"):
            raise PlanError(f"{c.label}: mapper must be 'incremental' or 'global'")
        if c.masks and plan.dataset.masks is None:
            raise PlanError(f"{c.label}: masks = true but [dataset] has no 'masks' folder")
    dependents: list[OpenMVSRun | BrushRun] = [*plan.openmvs, *plan.brush]
    for dep in dependents:
        plan.colmap_run(dep.source)
    for o in plan.openmvs:
        if o.texture_export not in ("ply", "obj"):
            raise PlanError(f"{o.label}: texture_export must be 'ply' or 'obj'")
        if o.masks and plan.dataset.masks is None:
            raise PlanError(f"{o.label}: masks = true but [dataset] has no 'masks' folder")


def config_dict(run: ColmapRun | OpenMVSRun | BrushRun) -> dict[str, Any]:
    return dataclasses.asdict(run)
