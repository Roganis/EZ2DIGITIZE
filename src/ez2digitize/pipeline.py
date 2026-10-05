# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mesh pipeline: chains the backend stages over a project's captures.

    sparse: features -> matching -> mapping -> undistort
            [-> mask-undistort]                                  (COLMAP)
    dense:  mvs-import -> densify -> mesh [-> refine] -> texture (OpenMVS)

The two halves run separately because the user adjusts the crop box on the
sparse result before densifying (Phase 2). `run_mesh` runs both.

Every stage goes through `core.stage.run_stage`, so an unchanged stage is
reused and a changed one invalidates everything after it. Stages run one at
a time, and only one pipeline runs per process: the 8 GB M1 can't fit two
reconstructions. Events are delivered on the calling thread; the UI runs the
pipeline on a worker thread, the CLI on the main one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from ez2digitize.backends import brush, colmap, openmvs
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.core.capture import CaptureBundle, list_bundles
from ez2digitize.core.hardware import detect_gpus
from ez2digitize.core.project import Project
from ez2digitize.core.resources import GIB, available_memory, cpu_threads
from ez2digitize.core.runner import CancelToken
from ez2digitize.core.runner import Event as ProcessEvent
from ez2digitize.core.stage import StageManifest, StageSpec, load_manifest, run_stage
from ez2digitize.diagnosis import explain
from ez2digitize.export import ExportError, ExportFormat, export_mesh, export_notes, export_splat

SPARSE_STAGES = ("features", "matching", "mapping", "undistort", "mask-undistort")
DENSE_STAGES = ("mvs-import", "densify", "mesh", "refine", "texture")
STAGES = SPARSE_STAGES + DENSE_STAGES
SPLAT_STAGE = "splat"
# Every stage a project can have: the mesh path and the splat branch.
ALL_STAGES = (*STAGES, SPLAT_STAGE)

# Below this share of registered images the result is suspect: tell the user.
MIN_REGISTERED_SHARE = 0.5


@dataclass(frozen=True)
class Tools:
    colmap: colmap.Colmap
    openmvs: openmvs.OpenMVS
    # Only needed for splats; None if it isn't installed.
    brush: brush.Brush | None = None

    @classmethod
    def locate(cls) -> Tools:
        """Find the backends (raises backends.common.BackendMissing for COLMAP
        or OpenMVS; Brush is optional)."""
        try:
            splats: brush.Brush | None = brush.locate()
        except BackendMissing:
            splats = None
        return cls(colmap=colmap.locate(), openmvs=openmvs.locate(), brush=splats)


@dataclass(frozen=True)
class MeshSettings:
    features: colmap.FeatureOptions = field(default_factory=colmap.FeatureOptions)
    # None: sequential matching if every capture is a video, else exhaustive.
    matching: colmap.MatchOptions | None = None
    mapper: colmap.MapperOptions = field(default_factory=colmap.MapperOptions)
    undistort: colmap.UndistortOptions = field(default_factory=colmap.UndistortOptions)
    densify: openmvs.DensifyOptions = field(default_factory=openmvs.DensifyOptions)
    mesh: openmvs.MeshOptions = field(default_factory=openmvs.MeshOptions)
    # None skips RefineMesh (slow; worth it for fine detail).
    refine: openmvs.RefineOptions | None = None
    texture: openmvs.TextureOptions = field(default_factory=openmvs.TextureOptions)
    # Use the project's masks (feature extraction and densification) if it has any.
    use_masks: bool = True
    # Formats exported to exports/ after texturing; empty: no export.
    export_formats: tuple[ExportFormat, ...] = ("obj", "glb")
    # Stand the export upright, centred, on the ground (see orientation).
    align: bool = True
    # Splat training (run_splat only).
    splat: brush.SplatOptions = field(default_factory=brush.SplatOptions)


# --- events --------------------------------------------------------------------


@dataclass(frozen=True)
class StageStarted:
    stage: str
    index: int  # 1-based position in the stages this call runs
    count: int


@dataclass(frozen=True)
class StageOutput:
    """An event from the stage's process (started, output line, progress)."""

    stage: str
    event: ProcessEvent


@dataclass(frozen=True)
class StageFinished:
    stage: str
    manifest: StageManifest
    reused: bool


@dataclass(frozen=True)
class Notice:
    """Something the user should know that doesn't stop the pipeline."""

    message: str


PipelineEvent = StageStarted | StageOutput | StageFinished | Notice
PipelineFunction = Callable[..., "MeshResult | SplatResult"]
PipelineHandler = Callable[[PipelineEvent], None]


# --- errors and results -----------------------------------------------------------


class PipelineError(Exception):
    """The pipeline can't start or can't go on (no images, no model...)."""


class StageFailed(PipelineError):
    """A stage's tool failed; `explanation` says why in plain words, when known."""

    def __init__(self, manifest: StageManifest, log: Path, tail: list[str]) -> None:
        self.explanation = explain(manifest.stage, manifest.exit_code, tail)
        where = f"stage {manifest.stage} failed (exit code {manifest.exit_code})"
        if self.explanation is None:
            message = where
        else:
            message = f"{self.explanation.title} ({where}). {self.explanation.advice}"
        super().__init__(message)
        self.manifest = manifest
        self.log = log
        self.tail = tail


class PipelineCancelled(PipelineError):
    pass


class PipelineBusy(PipelineError):
    pass


@dataclass
class SparseResult:
    model: Path
    registered_images: int
    total_images: int
    undistorted: StageManifest
    # Masks warped to the undistorted images, for OpenMVS; None if unmasked.
    masks: StageManifest | None = None


@dataclass
class MeshResult:
    sparse: SparseResult
    textured: StageManifest
    files: list[Path]
    # The exported files (OBJ, MTL, textures, GLB...), if any were asked for.
    exports: list[Path] = field(default_factory=list)


@dataclass
class SplatResult:
    sparse: SparseResult
    splat: StageManifest
    file: Path
    exports: list[Path] = field(default_factory=list)


# --- running -------------------------------------------------------------------

_running = threading.Lock()


@contextmanager
def _exclusive() -> Iterator[None]:
    if not _running.acquire(blocking=False):
        raise PipelineBusy("another reconstruction is already running")
    try:
        yield
    finally:
        _running.release()


class _Run:
    """Runs the stages of one call: numbering, events, failure handling."""

    def __init__(
        self,
        project: Project,
        stages: tuple[str, ...],
        on_event: PipelineHandler | None,
        cancel: CancelToken | None,
        force_from: str | None,
    ) -> None:
        if force_from is not None and force_from not in ALL_STAGES:
            raise PipelineError(f"unknown stage {force_from!r}; stages are {', '.join(ALL_STAGES)}")
        self.project = project
        self.stages = stages
        self.emit = on_event or (lambda _event: None)
        self.cancel = cancel
        self.force_from = force_from

    def __call__(self, spec: StageSpec) -> StageManifest:
        if self.cancel is not None and self.cancel.cancelled:
            raise PipelineCancelled("cancelled")
        stage = spec.name
        self.emit(StageStarted(stage, self.stages.index(stage) + 1, len(self.stages)))
        run = run_stage(
            self.project,
            spec,
            on_event=lambda event: self.emit(StageOutput(stage, event)),
            cancel=self.cancel,
            force=self.force_from == stage,
        )
        self.emit(StageFinished(stage, run.manifest, run.reused))
        if run.manifest.status == "cancelled":
            raise PipelineCancelled(f"cancelled during {stage}")
        if not run.manifest.succeeded:
            log = self.project.stage_dir(stage) / "log.txt"
            raise StageFailed(run.manifest, log, _tail(log))
        return run.manifest


def run_sparse(
    project: Project,
    tools: Tools,
    settings: MeshSettings | None = None,
    *,
    on_event: PipelineHandler | None = None,
    cancel: CancelToken | None = None,
    force_from: str | None = None,
) -> SparseResult:
    """Camera poses and undistorted images from all of the project's captures."""
    with _exclusive():
        return _sparse(project, tools, settings or MeshSettings(), on_event, cancel, force_from)


def run_dense(
    project: Project,
    tools: Tools,
    settings: MeshSettings | None = None,
    *,
    on_event: PipelineHandler | None = None,
    cancel: CancelToken | None = None,
    force_from: str | None = None,
) -> MeshResult:
    """The textured mesh from the sparse result already in the project."""
    settings = settings or MeshSettings()
    with _exclusive():
        sparse = _existing_sparse(project, settings)
        return _dense(project, tools, settings, sparse, on_event, cancel, force_from)


def run_mesh(
    project: Project,
    tools: Tools,
    settings: MeshSettings | None = None,
    *,
    on_event: PipelineHandler | None = None,
    cancel: CancelToken | None = None,
    force_from: str | None = None,
) -> MeshResult:
    """Both halves in one go, without stopping for the crop box."""
    settings = settings or MeshSettings()
    with _exclusive():
        sparse = _sparse(project, tools, settings, on_event, cancel, force_from)
        return _dense(project, tools, settings, sparse, on_event, cancel, force_from)


def run_splat(
    project: Project,
    tools: Tools,
    settings: MeshSettings | None = None,
    *,
    on_event: PipelineHandler | None = None,
    cancel: CancelToken | None = None,
    force_from: str | None = None,
    allow_software_gpu: bool = False,
) -> SplatResult:
    """Camera poses (reused if unchanged), then Gaussian splats with Brush.

    Refuses to train on a software renderer (llvmpipe), which would take
    days, unless `allow_software_gpu` (tests, CI).
    """
    settings = settings or MeshSettings()
    if tools.brush is None:
        raise PipelineError(
            "splats need Brush, which wasn't found; set its location in the "
            "reconstruction tools settings"
        )
    gpus = detect_gpus()
    if not allow_software_gpu and not any(not g.is_cpu for g in gpus):
        raise PipelineError(
            "training splats needs a GPU with a Vulkan driver (or Metal on macOS); only a "
            "software renderer was found"
            if gpus
            else "no GPU with a Vulkan driver was found"
        )
    with _exclusive():
        masked = settings.use_masks and _has_masks(project.masks_dir)
        stages = (*(s for s in SPARSE_STAGES if masked or s != "mask-undistort"), SPLAT_STAGE)
        sparse = _sparse(project, tools, settings, on_event, cancel, force_from, stages=stages)
        run = _Run(project, stages, on_event, cancel, force_from)
        manifest = run(
            brush.train(tools.brush, project, sparse.undistorted, options=settings.splat)
        )
        file = project.stage_dir(SPLAT_STAGE) / brush.SPLAT_FILE
        if not file.is_file():
            raise PipelineError("Brush finished but wrote no splat file; see its log")
        exports: list[Path] = []
        if settings.export_formats:
            try:
                exports = export_splat(project)
            except ExportError as exc:
                raise PipelineError(str(exc)) from exc
            run.emit(Notice(f"exported to {exports[0].parent}"))
        return SplatResult(sparse=sparse, splat=manifest, file=file, exports=exports)


def _sparse(
    project: Project,
    tools: Tools,
    settings: MeshSettings,
    on_event: PipelineHandler | None,
    cancel: CancelToken | None,
    force_from: str | None,
    *,
    stages: tuple[str, ...] | None = None,
) -> SparseResult:
    bundles = list_bundles(project)
    if not bundles:
        raise PipelineError("the project has no captures; import photos or a video first")
    try:
        total = len(colmap.image_names(bundles))
    except BackendError as exc:
        raise PipelineError(str(exc)) from exc
    masks = project.masks_dir if settings.use_masks and _has_masks(project.masks_dir) else None
    stages = stages or _stages(settings, masked=masks is not None)
    run = _Run(project, stages, on_event, cancel, force_from)
    sfm = tools.colmap

    feature_options = settings.features
    if feature_options.threads is None:
        cpus, available = cpu_threads(), available_memory()
        threads = colmap.feature_threads(feature_options.max_image_size, available, cpus)
        feature_options = replace(feature_options, threads=threads)
        if threads < cpus:
            run.emit(
                Notice(
                    f"finding features with {threads} of {cpus} CPU threads, to stay within "
                    f"the {available / GIB:.1f} GB of free memory"
                )
            )
    features = run(
        colmap.extract_features(sfm, project, bundles, masks=masks, options=feature_options)
    )
    matching_options = settings.matching or _auto_matching(bundles)
    matching = run(colmap.match_features(sfm, project, features, options=matching_options))
    mapping = run(
        colmap.map_sparse(sfm, project, matching, total_images=total, options=settings.mapper)
    )
    sparse_dir = project.stage_dir("mapping") / "sparse"
    model = colmap.best_model(sparse_dir)
    if model is None:
        raise PipelineError(
            "COLMAP could not reconstruct any cameras: the photos probably overlap too "
            "little or the object has too little texture"
        )
    registered = colmap.registered_images(model)
    run.emit(Notice(f"{registered} of {total} images registered (model {model.name})"))
    if len(colmap.models(sparse_dir)) > 1:
        run.emit(
            Notice(
                f"the photos split into {len(colmap.models(sparse_dir))} separate groups; "
                f"using the largest. More overlap between the groups would join them."
            )
        )
    if registered < total * MIN_REGISTERED_SHARE:
        run.emit(
            Notice(f"only {registered} of {total} images were placed; the mesh may be partial")
        )

    undistorted = run(
        colmap.undistort(sfm, project, mapping, model=model, options=settings.undistort)
    )
    warped = None
    if masks is not None:
        try:
            spec = colmap.undistort_masks(
                sfm, project, mapping, model=model, masks=masks, options=settings.undistort
            )
        except BackendError as exc:
            run.emit(Notice(f"densifying without masks: {exc}"))
        else:
            warped = run(spec)
    return SparseResult(
        model=model,
        registered_images=registered,
        total_images=total,
        undistorted=undistorted,
        masks=warped,
    )


def _dense(
    project: Project,
    tools: Tools,
    settings: MeshSettings,
    sparse: SparseResult,
    on_event: PipelineHandler | None,
    cancel: CancelToken | None,
    force_from: str | None,
) -> MeshResult:
    masked = sparse.masks is not None
    if settings.export_formats and settings.texture.export_type != "ply":
        raise PipelineError("exporting needs the texture step's PLY output (export_type 'ply')")
    run = _Run(project, _stages(settings, masked=masked), on_event, cancel, force_from)
    mvs = tools.openmvs

    imported = run(openmvs.import_colmap(mvs, project, sparse.undistorted))
    dense = run(
        openmvs.densify(mvs, project, imported, masks=sparse.masks, options=settings.densify)
    )
    mesh = run(openmvs.reconstruct_mesh(mvs, project, dense, options=settings.mesh))
    if settings.refine is not None:
        mesh = run(openmvs.refine_mesh(mvs, project, dense, mesh, options=settings.refine))
    textured = run(openmvs.texture_mesh(mvs, project, dense, mesh, options=settings.texture))
    folder = project.stage_dir(textured.stage)
    files = sorted(
        p
        for p in folder.iterdir()
        if p.name.startswith("scene_textured") and p.suffix.lower() not in (".mvs", ".log")
    )
    exports: list[Path] = []
    if settings.export_formats:
        names = ", ".join(f.upper() for f in settings.export_formats)
        run.emit(Notice(f"exporting {names}"))
        try:
            exports = export_mesh(
                project, settings.export_formats, stage=textured.stage, align=settings.align
            )
        except ExportError as exc:
            raise PipelineError(str(exc)) from exc
        run.emit(Notice(f"exported to {exports[0].parent if exports else project.exports_dir}"))
        for note in export_notes(exports):
            run.emit(Notice(note))
    return MeshResult(sparse=sparse, textured=textured, files=files, exports=exports)


def _existing_sparse(project: Project, settings: MeshSettings) -> SparseResult:
    """The sparse result of an earlier run_sparse, from its manifests."""
    undistorted = load_manifest(project.stage_dir("undistort"))
    if undistorted is None or not undistorted.succeeded:
        raise PipelineError("run the sparse reconstruction first")
    warped = load_manifest(project.stage_dir("mask-undistort"))
    # Only masks made from the same mapping run as the undistorted images.
    if not (
        settings.use_masks
        and warped is not None
        and warped.succeeded
        and warped.inputs.get("mapping") == undistorted.inputs.get("mapping")
    ):
        warped = None
    model_name = undistorted.parameters.get("model")
    model = project.stage_dir("mapping") / "sparse" / str(model_name)
    total = len(colmap.image_names(list_bundles(project)))
    return SparseResult(
        model=model,
        registered_images=colmap.registered_images(model),
        total_images=total,
        undistorted=undistorted,
        masks=warped,
    )


def _stages(settings: MeshSettings, *, masked: bool) -> tuple[str, ...]:
    skip = set()
    if settings.refine is None:
        skip.add("refine")
    if not masked:
        skip.add("mask-undistort")
    return tuple(s for s in STAGES if s not in skip)


# Up to this many images, every pair is matched even for video: it closes the
# loop of an orbit, which sequential matching (without a vocabulary tree)
# can't. Pairs grow with the square: 63 images took 17 s on the reference
# desktop, 200 would take about 3 minutes.
EXHAUSTIVE_MAX_IMAGES = 200


def _auto_matching(bundles: list[CaptureBundle]) -> colmap.MatchOptions:
    images = sum(len(b.images) for b in bundles)
    # Beyond that, video frames are matched with their neighbours in time.
    if images > EXHAUSTIVE_MAX_IMAGES and all(b.source == "video" for b in bundles):
        return colmap.MatchOptions(mode="sequential")
    return colmap.MatchOptions(mode="exhaustive")


def _has_masks(masks_dir: Path) -> bool:
    return masks_dir.is_dir() and any(masks_dir.rglob("*.png"))


def _tail(log: Path, lines: int = 30) -> list[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []
