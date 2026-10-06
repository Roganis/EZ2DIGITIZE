# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mesh pipeline: chains the backend stages over a project's captures.

    sparse: features -> matching -> mapping -> undistort
            [-> mask-undistort]                                  (COLMAP)
    dense:  mvs-import -> densify -> mesh [-> refine] -> texture (OpenMVS)
    splat:  sparse, then splat                                   (Brush)

A plugin chosen for camera placement (see ez2digitize.plugins) runs as the
mapping stage instead of features, matching and mapping; one chosen for
splats runs as the splat stage instead of Brush.

The two halves run separately because the user adjusts the crop box on the
sparse result before densifying (Phase 2). `run_mesh` runs both.

Every stage goes through `core.stage.run_stage`, so an unchanged stage is
reused and a changed one invalidates everything after it. Stages run one at
a time, and only one pipeline runs per process: the 8 GB M1 can't fit two
reconstructions. Events are delivered on the calling thread; the UI runs the
pipeline on a worker thread, the CLI on the main one.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from ez2digitize import coverage, crop, markers, plugins, scale, sides, splat_mesh
from ez2digitize.backends import brush, colmap, colmap_model, openmvs
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.core import download, photos
from ez2digitize.core.capture import CaptureBundle, list_bundles
from ez2digitize.core.hardware import detect_gpus
from ez2digitize.core.meshio import MeshFormatError
from ez2digitize.core.photos import exif_orientations
from ez2digitize.core.project import Project
from ez2digitize.core.resources import GIB, available_memory, cpu_threads
from ez2digitize.core.runner import CancelToken
from ez2digitize.core.runner import Event as ProcessEvent
from ez2digitize.core.stage import StageManifest, StageSpec, load_manifest, run_stage
from ez2digitize.diagnosis import explain
from ez2digitize.export import ExportError, ExportFormat, export_mesh, export_notes, export_splat
from ez2digitize.masks import has_masks
from ez2digitize.motion import measured_downs
from ez2digitize.subject import Subject

SPARSE_STAGES = ("features", "matching", "mapping", "undistort", "mask-undistort")
DENSE_STAGES = ("mvs-import", "densify", "mesh", "refine", "texture")
STAGES = SPARSE_STAGES + DENSE_STAGES
SPLAT_STAGE = "splat"
SPLAT_MESH_STAGE = splat_mesh.STAGE  # a mesh from the splats (optional)
# Every stage a project can have: the mesh path and the splat branch.
ALL_STAGES = (*STAGES, SPLAT_STAGE, SPLAT_MESH_STAGE)

# Below this share of registered images the result is suspect: tell the user.
MIN_REGISTERED_SHARE = 0.5


@dataclass(frozen=True)
class Tools:
    colmap: colmap.Colmap
    openmvs: openmvs.OpenMVS
    # Only needed for splats; None if it isn't installed.
    brush: brush.Brush | None = None
    # Plugins chosen instead of COLMAP's camera placement and of Brush.
    poses: plugins.Plugin | None = None
    splats: plugins.Plugin | None = None

    def used_plugins(self) -> list[plugins.Plugin]:
        """The plugins these tools use."""
        return [p for p in (self.poses, self.splats) if p is not None]

    @classmethod
    def locate(cls) -> Tools:
        """Find the backends and the chosen plugins (raises backends.common.
        BackendMissing for COLMAP or OpenMVS, plugins.PluginError for a chosen
        plugin that can't be used; Brush is optional)."""
        try:
            splats: brush.Brush | None = brush.locate()
        except BackendMissing:
            splats = None
        return cls(
            colmap=colmap.locate(),
            openmvs=openmvs.locate(),
            brush=splats,
            poses=plugins.chosen("poses"),
            splats=plugins.chosen("splats"),
        )


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
    # A mesh from the splats too (run_splat), with these options; None: no.
    splat_mesh: splat_mesh.SplatMeshOptions | None = None
    # A scene skips the advice that assumes photos all round an object.
    subject: Subject = "object"


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
PipelineFunction = Callable[..., "SparseResult | MeshResult | SplatResult"]
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
    # The mesh from the splats (splat_mesh's mesh.ply), if asked for.
    mesh: Path | None = None
    mesh_exports: list[Path] = field(default_factory=list)


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
    if tools.brush is None and tools.splats is None:
        raise PipelineError(
            "splats need Brush, which wasn't found; set its location in the "
            "reconstruction tools settings"
        )
    gpus = detect_gpus() if tools.splats is None or tools.splats.gpu else []
    if (
        not allow_software_gpu
        and (tools.splats is None or tools.splats.gpu)
        and not any(not g.is_cpu for g in gpus)
    ):
        raise PipelineError(
            "training splats needs a GPU with a Vulkan driver (or Metal on macOS); only a "
            "software renderer was found"
            if gpus
            else "no GPU with a Vulkan driver was found"
        )
    with _exclusive():
        masked = settings.use_masks and has_masks(project)
        stages = (*_sparse_stages(tools, masked=masked), SPLAT_STAGE)
        if settings.splat_mesh is not None:
            stages = (*stages, SPLAT_MESH_STAGE)
        sparse = _sparse(project, tools, settings, on_event, cancel, force_from, stages=stages)
        run = _Run(project, stages, on_event, cancel, force_from)
        masks = sparse.masks
        if masks is not None and (clash := _stem_clash(project)):
            run.emit(Notice(f"training splats without masks: {clash}"))
            masks = None
        if tools.splats is not None:
            run.emit(Notice(f"training splats with {plugins.describe(tools.splats)}"))
            spec = _plugin_spec(
                lambda p: plugins.splat_stage(
                    p,
                    project,
                    sparse.undistorted,
                    sfm=tools.colmap,
                    options=settings.splat,
                    masks=masks,
                ),
                tools.splats,
            )
        else:
            assert tools.brush is not None
            spec = brush.train(
                tools.brush, project, sparse.undistorted, options=settings.splat, masks=masks
            )
        manifest = run(spec)
        file = project.stage_dir(SPLAT_STAGE) / brush.SPLAT_FILE
        if not file.is_file():
            trainer = tools.splats.name if tools.splats is not None else "Brush"
            raise PipelineError(f"{trainer} finished but wrote no splat file; see its log")
        exports: list[Path] = []
        if settings.export_formats:
            try:
                exports = export_splat(project)
            except ExportError as exc:
                raise PipelineError(str(exc)) from exc
            run.emit(Notice(f"exported to {exports[0].parent}"))
        result = SplatResult(sparse=sparse, splat=manifest, file=file, exports=exports)
        if settings.splat_mesh is not None:
            _splat_mesh(project, tools, settings, sparse, manifest, run, result)
        return result


def _splat_mesh(
    project: Project,
    tools: Tools,
    settings: MeshSettings,
    sparse: SparseResult,
    splats: StageManifest,
    run: _Run,
    result: SplatResult,
) -> None:
    """The mesh from the splats (see splat_mesh), exported like the splats."""
    options = settings.splat_mesh or splat_mesh.SplatMeshOptions()
    spec = splat_mesh.mesh_stage(tools.colmap, project, splats, sparse.undistorted, options=options)
    try:
        run(spec)
    except splat_mesh.SplatMeshError as exc:
        raise PipelineError(f"no mesh from the splats: {exc}") from exc
    mesh = project.stage_dir(SPLAT_MESH_STAGE) / splat_mesh.MESH
    try:
        faces = splat_mesh.read_mesh(mesh).faces
    except (OSError, MeshFormatError):
        faces = None
    if faces is None or not len(faces):
        raise PipelineError(
            "the mesh from the splats came out empty: the splats are too sparse for the "
            "trimming (try a lower trim value, or more training steps)"
        )
    result.mesh = mesh
    if settings.export_formats:
        try:
            result.mesh_exports = splat_mesh.export(project)
        except ExportError as exc:
            raise PipelineError(str(exc)) from exc
        run.emit(Notice(f"mesh from the splats exported to {result.mesh_exports[0].parent}"))


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
    masks = project.masks_dir if settings.use_masks and has_masks(project) else None
    stages = stages or _stages(settings, tools, masked=masks is not None)
    run = _Run(project, stages, on_event, cancel, force_from)
    for problem in sides.check(project, bundles, use_masks=settings.use_masks):
        run.emit(Notice(f"two-sided scan: {problem}"))
    sfm = tools.colmap
    if tools.poses is not None:
        mapping = _plugin_mapping(project, tools, tools.poses, bundles, masks, run)
    else:
        mapping = _colmap_mapping(project, tools, settings, bundles, masks, run, total)
    sparse_dir = project.stage_dir("mapping") / "sparse"
    model = colmap.best_model(sparse_dir)
    if model is None:
        raise PipelineError(
            (
                f"{tools.poses.name} placed no cameras (or wrote no COLMAP model in "
                "sparse/0; see its log)"
            )
            if tools.poses is not None
            else "COLMAP could not reconstruct any cameras: the photos probably overlap too "
            "little or the object has too little texture"
        )
    registered = colmap.registered_images(model)
    run.emit(Notice(f"{registered} of {total} images registered (model {model.name})"))
    if (two_sided := sides.sides(bundles)) is not None and two_sided.complete:
        try:
            placed = colmap_model.read_images(model)
        except BackendError as exc:
            run.emit(Notice(f"could not check how the two sides joined: {exc}"))
        else:
            run.emit(Notice(sides.join_notice(sides.joined(two_sided, placed))))
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
    for finding in _coverage_notes(project, model, bundles, settings.subject):
        run.emit(Notice(finding))

    undistorted = run(
        colmap.undistort(sfm, project, mapping, model=model, options=settings.undistort)
    )
    if (note := markers.auto_scale(project)) is not None:
        run.emit(Notice(note))
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


def _plugin_mapping(
    project: Project,
    tools: Tools,
    plugin: plugins.Plugin,
    bundles: list[CaptureBundle],
    masks: Path | None,
    run: _Run,
) -> StageManifest:
    run.emit(Notice(f"placing the cameras with {plugins.describe(plugin)}"))
    if masks is not None and not plugin.with_masks:
        run.emit(Notice(f"{plugin.name} doesn't use masks; the dense cloud still does"))
    return run(
        _plugin_spec(
            lambda p: plugins.pose_stage(p, project, bundles, sfm=tools.colmap, masks=masks),
            plugin,
        )
    )


def _plugin_spec(build: Callable[[plugins.Plugin], StageSpec], plugin: plugins.Plugin) -> StageSpec:
    try:
        return build(plugin)
    except BackendError as exc:
        raise PipelineError(str(exc)) from exc


def _colmap_mapping(
    project: Project,
    tools: Tools,
    settings: MeshSettings,
    bundles: list[CaptureBundle],
    masks: Path | None,
    run: _Run,
    total: int,
) -> StageManifest:
    """COLMAP's camera placement: features, matching, mapping."""
    sfm = tools.colmap
    feature_options = settings.features
    lightglue = None
    if feature_options.kind == "aliked":
        model = _fetch(colmap.ALIKED_MODEL, "the ALIKED model", run)
        lightglue = _fetch(colmap.LIGHTGLUE_MODEL, "the LightGlue model", run)
        if model is None or lightglue is None:
            raise PipelineError(
                "learned features need the ALIKED and LightGlue models, which could not be "
                "downloaded (see the notes above); try again, or use SIFT features"
            )
        feature_options = replace(feature_options, model=model)
    for bundle in bundles:  # usually done at import; quick
        photos.inspect_bundle(bundle)
    groups = photos.camera_groups(bundles)
    if groups and feature_options.camera_grouping == "per_capture":
        # COLMAP would calibrate a capture as one camera: extract a camera per
        # photo, then merge them by camera and zoom setting before matching.
        feature_options = replace(feature_options, camera_grouping="per_image")
        count = len(set(groups.values()))
        run.emit(Notice(f"calibrating {count} cameras or zoom settings separately"))
    else:
        groups = {}
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
    matching_options = _matching(settings.matching, bundles, run, feature_options.kind)
    matching_options = replace(matching_options, features=feature_options.kind, lightglue=lightglue)
    matching = run(
        colmap.match_features(
            sfm, project, features, options=matching_options, camera_groups=groups or None
        )
    )
    return run(
        colmap.map_sparse(sfm, project, matching, total_images=total, options=settings.mapper)
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
    run = _Run(project, _stages(settings, tools, masked=masked), on_event, cancel, force_from)
    mvs = tools.openmvs

    imported = run(openmvs.import_colmap(mvs, project, sparse.undistorted))
    box = crop.current(project)
    if box is None and crop.stored(project) is not None:
        run.emit(
            Notice(
                "the crop box was drawn on an earlier camera placement, so it is not used; "
                "set it again in the 3D view"
            )
        )
    dense = run(
        openmvs.densify(
            mvs,
            project,
            imported,
            masks=sparse.masks,
            options=settings.densify,
            roi=box.roi_text() if box is not None else None,
        )
    )
    if box is not None:
        run.emit(Notice("the dense cloud keeps what is inside the crop box"))
    if scale.current(project) is None and scale.stored(project) is not None:
        run.emit(
            Notice(
                "the scale was set on an earlier camera placement, so exports are in "
                "arbitrary units; set it again in the 3D view"
            )
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


def _coverage_notes(
    project: Project, model: Path, bundles: list[CaptureBundle], subject: Subject
) -> list[str]:
    """What the camera placement says about the capture (advice only)."""
    notes = []
    try:
        placed = colmap_model.read_images(model)
        unplaced = sorted(set(colmap.image_names(bundles)) - set(placed))
        if unplaced:
            notes.append(
                f"not placed: {coverage.name_list(unplaced)}. They overlap too little with "
                "the others, or are blurry or of something else."
            )
        database = colmap.matched_database(project)
        weak = [n for n in coverage.weak_photos(database) if n in placed] if database else []
        if weak:
            notes.append(
                f"few matches with the others: {coverage.name_list(weak)}. More photos "
                "between them and their neighbours would make the result more reliable."
            )
        if subject == "object":  # all round an object; a room or a street isn't
            analysis = coverage.analyse(
                model,
                exif_orientations(bundles),
                sides.upright_names(bundles),
                measured_downs(bundles),
            )
            notes += analysis.findings if analysis else ()
    except (OSError, ValueError, BackendError, sqlite3.Error):
        pass  # advice only: never stop the run for it
    return notes


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


def _stem_clash(project: Project) -> str | None:
    """Why Brush can't tell two photos' masks apart, if it can't.

    Brush finds a mask by the image's file stem, ignoring case, in one folder
    for all captures (OpenMVS has the same limit, but compares case).
    """
    seen: dict[str, str] = {}
    for name in colmap.image_names(list_bundles(project)):
        stem = Path(name).stem.lower()
        if stem in seen:
            return f"{seen[stem]} and {name} have the same file name apart from case"
        seen[stem] = name
    return None


def _sparse_stages(tools: Tools, *, masked: bool) -> tuple[str, ...]:
    skip = set()
    if tools.poses is not None:
        skip |= {"features", "matching"}  # the plugin is the mapping stage
    if not masked:
        skip.add("mask-undistort")
    return tuple(s for s in SPARSE_STAGES if s not in skip)


def _stages(settings: MeshSettings, tools: Tools, *, masked: bool) -> tuple[str, ...]:
    dense = tuple(s for s in DENSE_STAGES if settings.refine is not None or s != "refine")
    return _sparse_stages(tools, masked=masked) + dense


# Up to this many images, every pair is matched, whatever the capture: it is
# the most thorough. Pairs grow with the square: 63 images took 17 s on the
# reference desktop, 200 would take about 3 minutes, 1000 over an hour.
EXHAUSTIVE_MAX_IMAGES = 200
# Beyond that, photos with GPS positions are paired by distance if at least
# this share has one (outdoors, from a phone).
GPS_SHARE = 0.9


def _matching(
    chosen: colmap.MatchOptions | None,
    bundles: list[CaptureBundle],
    run: _Run,
    features: colmap.FeatureKind = "sift",
) -> colmap.MatchOptions:
    """The matching settings: chosen ones (with the vocabulary tree they need), or
    picked from the photos (see colmap.MatchOptions for the modes).

    Beyond EXHAUSTIVE_MAX_IMAGES: video frames sequentially, with loops found
    by the vocabulary tree; photos by GPS if they have it, else by the tree,
    else sequentially in name order (the order they were taken).
    """
    if chosen is not None:
        if chosen.mode == "vocab_tree" and chosen.vocab_tree is None:
            tree = _vocab_tree(run, features)
            if tree is None:
                raise PipelineError("vocabulary tree matching needs COLMAP's vocabulary tree")
            return replace(chosen, vocab_tree=tree)
        return chosen
    images = sum(len(b.images) for b in bundles)
    if images <= EXHAUSTIVE_MAX_IMAGES:
        return colmap.MatchOptions(mode="exhaustive")
    if all(b.source == "video" for b in bundles):
        return colmap.MatchOptions(mode="sequential", vocab_tree=_vocab_tree(run, features))
    if photos.gps_share(bundles) >= GPS_SHARE:
        run.emit(Notice(f"{images} photos with GPS positions: each is matched with its neighbours"))
        return colmap.MatchOptions(mode="spatial")
    tree = _vocab_tree(run, features)
    if tree is not None:
        run.emit(Notice(f"{images} photos: each is matched with the most similar ones"))
        return colmap.MatchOptions(mode="vocab_tree", vocab_tree=tree)
    run.emit(
        Notice(
            f"{images} photos: each is matched with those taken just before and after it, "
            "so the photos should be taken in order, without jumping around"
        )
    )
    return colmap.MatchOptions(mode="sequential")


def _vocab_tree(run: _Run, features: colmap.FeatureKind) -> Path | None:
    """COLMAP's vocabulary tree for these features, downloaded the first time."""
    return _fetch(colmap.VOCAB_TREES[features], "COLMAP's vocabulary tree", run)


def _fetch(pinned: colmap.Pinned, what: str, run: _Run) -> Path | None:
    """A file COLMAP would download (see colmap.Pinned), fetched the first time;
    None, with a notice, if that fails."""
    found = colmap.find_pinned(pinned)
    if found is not None:
        return found
    run.emit(Notice(f"downloading {what} ({pinned.size / 1e6:.0f} MB, once)"))
    try:
        return colmap.fetch_pinned(pinned, cancel=run.cancel)
    except download.DownloadCancelled as exc:
        raise PipelineCancelled("cancelled") from exc
    except download.DownloadError as exc:
        run.emit(Notice(f"could not download {what}: {exc}"))
        return None


def _tail(log: Path, lines: int = 30) -> list[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []
