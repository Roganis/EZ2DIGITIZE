# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless command line: create a project, import captures, run the pipeline.

    ez2d new ~/scans/skull
    ez2d import ~/scans/skull ~/Pictures/skull-photos
    ez2d import ~/scans/skull ~/Videos/skull.mp4 --frames 120
    ez2d photos ~/scans/skull               # photo checks
    ez2d photos ~/scans/skull --exclude Preview.jpg
    ez2d masks ~/scans/skull                # automatic masks, then a review
    ez2d masks ~/scans/skull --drop IMG_0012.JPG
    ez2d import ~/scans/skull ~/Pictures/skull-under --flipped   # the other side
    ez2d run ~/scans/skull                  # everything
    ez2d run ~/scans/skull --sparse-only    # stop before densifying
    ez2d crop ~/scans/skull --auto          # crop box around the sparse points
    ez2d scale ~/scans/skull --distance 42  # the picked points are 42 mm apart
    ez2d orient ~/scans/skull --tilt x      # it lay on its side: a quarter turn
    ez2d status ~/scans/skull
    ez2d export ~/scans/skull --formats glb # again, e.g. in other formats
    ez2d check                              # which COLMAP and OpenMVS are used

The same code the GUI uses; this is how regression datasets run on the
reference machines and in CI.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO, cast

import segno

from ez2digitize import (
    crop,
    diagnostics,
    licenses,
    masks,
    presets,
    scale,
    sides,
    upright,
    video,
    views,
)
from ez2digitize.backends import brush, colmap, ffmpeg, openmvs
from ez2digitize.backends.common import BackendError, bundled_bin_dir
from ez2digitize.core import hardware, photos
from ez2digitize.core.capture import (
    CaptureBundle,
    CaptureError,
    classify,
    import_folder,
    list_bundles,
)
from ez2digitize.core.project import Project, ProjectError
from ez2digitize.core.runner import CancelToken, Output, Progress
from ez2digitize.core.stage import load_manifest
from ez2digitize.export import FORMATS, ExportError, ExportFormat, export_mesh, export_notes
from ez2digitize.pipeline import (
    ALL_STAGES,
    MeshResult,
    MeshSettings,
    Notice,
    PipelineCancelled,
    PipelineError,
    PipelineEvent,
    SparseResult,
    SplatResult,
    StageFailed,
    StageFinished,
    StageOutput,
    StageStarted,
    Tools,
    run_dense,
    run_mesh,
    run_sparse,
    run_splat,
)
from ez2digitize.upload import UploadSession
from ez2digitize.watch import SETTLE_S, FolderWatch, import_ready


def commands() -> tuple[str, ...]:
    """The CLI's command names (the packaged app's launcher dispatches on them)."""
    sub = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    return tuple(sub.choices)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except (
        ProjectError,
        CaptureError,
        BackendError,
        PipelineError,
        ExportError,
        masks.MaskingError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ez2d", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True, metavar="COMMAND")

    new = sub.add_parser("new", help="create an empty project folder")
    new.add_argument("project", type=Path)
    new.add_argument("--name", help="display name (default: the folder name)")
    new.set_defaults(func=_cmd_new)

    imp = sub.add_parser("import", help="import folders of photos, or videos, as capture bundles")
    imp.add_argument("project", type=Path)
    imp.add_argument("sources", type=Path, nargs="+", metavar="FOLDER_OR_VIDEO")
    imp.add_argument(
        "--masks",
        type=Path,
        help="folder of masks in COLMAP naming (<image name>.png), for a single folder",
    )
    imp.add_argument(
        "--frames",
        type=int,
        default=video.DEFAULT_FRAMES,
        help=f"frames to keep from each video (default {video.DEFAULT_FRAMES})",
    )
    imp.add_argument("--ffmpeg", type=Path, help="FFmpeg executable")
    imp.add_argument(
        "--flipped",
        action="store_true",
        help="the object is turned over in these photos (the second side of a two-sided scan)",
    )
    imp.set_defaults(func=_cmd_import)

    flip = sub.add_parser(
        "flip",
        help="mark a capture as the turned-over side of a two-sided scan (or undo it)",
    )
    flip.add_argument("project", type=Path)
    flip.add_argument("capture", help="capture id (see `ez2d status`)")
    flip.add_argument("--undo", action="store_true", help="mark it as the first side again")
    flip.set_defaults(func=_cmd_flip)

    phone = sub.add_parser(
        "upload", help="receive photos from a phone over Wi-Fi (prints a QR code to scan)"
    )
    phone.add_argument("project", type=Path)
    phone.add_argument("--port", type=int, default=0, help="port to listen on (default: any)")
    phone.set_defaults(func=_cmd_upload)

    synced = sub.add_parser(
        "watch",
        help="import the photos a phone syncs to a folder (Syncthing, iCloud Drive...) "
        "once they have all arrived",
    )
    synced.add_argument("project", type=Path)
    synced.add_argument("folder", type=Path, help="the folder the phone's photos sync to")
    synced.add_argument(
        "--since",
        type=float,
        metavar="MINUTES",
        help="also take the photos already there from the last MINUTES (default: only new ones)",
    )
    synced.add_argument(
        "--settle",
        type=float,
        default=SETTLE_S,
        metavar="SECONDS",
        help=f"import once nothing new has arrived for this long (default {SETTLE_S:.0f})",
    )
    synced.set_defaults(func=_cmd_watch)

    checks = sub.add_parser(
        "photos", help="check the photos; leave some out of the reconstruction or bring them back"
    )
    checks.add_argument("project", type=Path)
    checks.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="PHOTO",
        help="photos to leave out: file name, or <capture id>/<file name> if ambiguous",
    )
    checks.add_argument("--include", nargs="+", default=[], metavar="PHOTO", help="bring back")
    checks.set_defaults(func=_cmd_photos)

    mask = sub.add_parser(
        "masks",
        help="separate the object from the background (automatic masks), review them",
        description="Without options: make automatic masks for the photos that have none "
        "(the model, 179 MB, is downloaded the first time), then list the masks to look at.",
    )
    mask.add_argument("project", type=Path)
    action = mask.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="only list the masks")
    action.add_argument("--drop", nargs="+", metavar="PHOTO", help="use these photos whole")
    action.add_argument("--restore", nargs="+", metavar="PHOTO", help="use their masks again")
    action.add_argument("--clear", action="store_true", help="remove the automatic masks")
    action.add_argument(
        "--import",
        dest="import_folder",
        type=Path,
        metavar="FOLDER",
        help="import a folder of masks (<image name>.png) for --capture",
    )
    mask.add_argument("--capture", help="capture id for --import (default: the only one)")
    mask.add_argument("--force", action="store_true", help="make the automatic masks again")
    mask.set_defaults(func=_cmd_masks)

    box = sub.add_parser(
        "crop",
        help="the crop box: what the dense reconstruction keeps",
        description="Without options: show the crop box. Coordinates are in the upright "
        "frame the 3D view shows (Y up), after camera placement (`ez2d run --sparse-only`).",
    )
    box.add_argument("project", type=Path)
    change = box.add_mutually_exclusive_group()
    change.add_argument(
        "--auto", action="store_true", help="a box around most of the sparse points"
    )
    change.add_argument(
        "--set",
        nargs=6,
        type=float,
        metavar=("CX", "CY", "CZ", "HX", "HY", "HZ"),
        help="centre and half sizes",
    )
    change.add_argument("--clear", action="store_true", help="no box: OpenMVS's own estimate")
    box.add_argument("--yaw", type=float, default=0.0, help="degrees about the vertical axis")
    box.set_defaults(func=_cmd_crop)

    size = sub.add_parser(
        "scale",
        help="real-world size: the real distance between two points",
        description="Without options: show the scale. The two points are usually picked in "
        "the GUI's 3D view; here they are given in the upright frame it shows (Y up), "
        "after camera placement. With the scale set, exports are in millimetres (STL, "
        "3MF) and metres (OBJ, GLB, point cloud).",
    )
    size.add_argument("project", type=Path)
    size.add_argument(
        "--points",
        nargs=6,
        type=float,
        metavar=("AX", "AY", "AZ", "BX", "BY", "BZ"),
        help="the two points (needs --distance)",
    )
    size.add_argument(
        "--distance", type=float, metavar="MM", help="their real distance, in millimetres"
    )
    size.add_argument("--clear", action="store_true", help="no scale: arbitrary units")
    size.set_defaults(func=_cmd_scale)

    orient = sub.add_parser(
        "orient",
        help="which way the model stands (the export's up and facing)",
        description="Without options: show the orientation. By default it comes from how "
        "the photos were held; these correct it, after camera placement. Points are in the "
        "upright frame the 3D view shows (Y up). A crop box is re-fitted level.",
    )
    orient.add_argument("project", type=Path)
    how = orient.add_mutually_exclusive_group()
    how.add_argument("--auto", action="store_true", help="back to the estimate from the photos")
    how.add_argument(
        "--level",
        nargs=9,
        type=float,
        metavar="V",
        help="three points (x y z each) on the surface the object stands on",
    )
    how.add_argument(
        "--tilt",
        choices=("x", "-x", "z", "-z"),
        help="a quarter turn about a horizontal axis (- turns the other way)",
    )
    orient.add_argument("--turn", type=float, metavar="DEG", help="degrees about the vertical")
    orient.set_defaults(func=_cmd_orient)

    run = sub.add_parser("run", help="reconstruct a textured mesh")
    run.add_argument("project", type=Path)
    part = run.add_mutually_exclusive_group()
    part.add_argument("--sparse-only", action="store_true", help="stop after camera poses")
    part.add_argument("--dense-only", action="store_true", help="only the OpenMVS stages")
    part.add_argument(
        "--splat", action="store_true", help="Gaussian splats with Brush instead of a mesh"
    )
    run.add_argument("--brush", type=Path, help="Brush executable (brush_app)")
    run.add_argument("--steps", type=int, help="splat training steps (default from --quality)")
    run.add_argument(
        "--allow-software-gpu",
        action="store_true",
        help="train splats even on a software renderer (llvmpipe; very slow, for testing)",
    )
    run.add_argument(
        "--quality",
        choices=presets.QUALITIES,
        help="preset: fast, balanced or high (default: the project's last, else balanced); "
        "the options below override single values",
    )
    run.add_argument("--max-image-size", type=int, help="COLMAP feature image size")
    run.add_argument("--mapper", choices=["global", "incremental"], default="global")
    run.add_argument("--level", type=int, help="OpenMVS resolution level (0 = full size)")
    run.add_argument(
        "--refine", action=argparse.BooleanOptionalAction, help="run RefineMesh (slow)"
    )
    run.add_argument(
        "--faces", type=int, help="simplify the mesh to about this many faces before texturing"
    )
    run.add_argument(
        "--export",
        type=_formats,
        default=("obj", "glb"),
        metavar="FORMATS",
        help="formats to export, comma-separated: obj, glb, ply, stl, 3mf (printing, no "
        "texture), points (dense point cloud), or none (default obj,glb)",
    )
    run.add_argument("--no-masks", action="store_true", help="ignore the project's masks")
    run.add_argument(
        "--no-align", action="store_true", help="export in the reconstruction's own frame"
    )
    run.add_argument("--threads", type=int, help="limit CPU threads of every tool")
    run.add_argument("--force-from", choices=ALL_STAGES, help="re-run this stage and all after it")
    run.add_argument("--colmap", type=Path, help="COLMAP executable")
    run.add_argument("--openmvs-dir", type=Path, help="folder with the OpenMVS tools")
    run.add_argument("-v", "--verbose", action="store_true", help="print the tools' output")
    run.set_defaults(func=_cmd_run)

    export = sub.add_parser("export", help="export the textured mesh again")
    export.add_argument("project", type=Path)
    export.add_argument(
        "--formats",
        type=_formats,
        default=("obj", "glb"),
        metavar="FORMATS",
        help="comma-separated: obj, glb, ply, stl, 3mf, points (default obj,glb)",
    )
    export.add_argument(
        "--no-align", action="store_true", help="keep the reconstruction's own frame"
    )
    export.set_defaults(func=_cmd_export)

    check = sub.add_parser("check", help="find the reconstruction tools and check they run")
    check.add_argument("--colmap", type=Path, help="COLMAP executable")
    check.add_argument("--openmvs-dir", type=Path, help="folder with the OpenMVS tools")
    check.add_argument("--ffmpeg", type=Path, help="FFmpeg executable (for video import)")
    check.add_argument("--brush", type=Path, help="Brush executable (for splats)")
    check.set_defaults(func=_cmd_check)

    diag = sub.add_parser(
        "diagnostics", help="zip the logs and settings (not the photos) for a bug report"
    )
    diag.add_argument("project", type=Path)
    diag.add_argument("-o", "--output", type=Path, help="zip file (default: in the current folder)")
    diag.set_defaults(func=_cmd_diagnostics)

    lic = sub.add_parser("licenses", help="third-party components, licenses and sources")
    lic.add_argument("--gpl", action="store_true", help="print EZ2DIGITIZE's own license")
    lic.set_defaults(func=_cmd_licenses)

    status = sub.add_parser("status", help="show captures and stage results")
    status.add_argument("project", type=Path)
    status.set_defaults(func=_cmd_status)
    return parser


def _cmd_new(args: argparse.Namespace) -> int:
    project = Project.create(args.project, name=args.name)
    print(f"created project {project.name!r} in {project.root}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    if args.masks is not None and (len(args.sources) != 1 or not args.sources[0].is_dir()):
        print("error: --masks needs exactly one folder to import", file=sys.stderr)
        return 2
    if args.frames < 1:
        print("error: --frames must be at least 1", file=sys.stderr)
        return 2
    project = Project.open(args.project)
    for source in args.sources:
        if source.is_dir():
            bundle, skipped = import_folder(project, source, flipped=args.flipped)
            side = " (turned over)" if args.flipped else ""
            print(f"{source}: {len(bundle.files)} photos -> capture {bundle.id}{side}")
            for path in skipped:
                why = "a video: import it on its own" if classify(path) else "not a photo"
                print(f"  skipped ({why}): {path.name}")
            if args.masks is not None:
                copied = masks.import_folder_masks(project, bundle, args.masks)
                print(f"  {len(copied)} of {len(bundle.images)} masks imported")
        else:
            bundle = _import_video(project, source, args)
            if args.flipped:
                bundle.set_flipped(True)
            info = bundle.source_info
            print(
                f"{source}: {info['frames']} frames, the sharpest of {info['candidates']} "
                f"extracted -> capture {bundle.id}"
            )
    _check_photos(project)
    _print_sides(project)
    return 0


def _cmd_flip(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    bundles = list_bundles(project)
    bundle = _one_capture(bundles, args.capture)
    bundle.set_flipped(not args.undo)
    side = "the first side" if args.undo else "turned over"
    print(f"capture {bundle.id}: {side}")
    _print_sides(project)
    return 0


def _print_sides(project: Project) -> None:
    """The state of a two-sided scan, if this is one."""
    bundles = list_bundles(project)
    found = sides.sides(bundles)
    if found is None:
        return
    print(
        f"two-sided scan: {len(found.first)} photos of the first side, "
        f"{len(found.turned)} turned over"
    )
    for problem in sides.check(project, bundles, use_masks=True):
        print(f"  to do: {problem}")


def _import_video(project: Project, path: Path, args: argparse.Namespace) -> CaptureBundle:
    tool = ffmpeg.locate(args.ffmpeg)
    if not tool.supported:
        print(f"note: FFmpeg {tool.version} is older than {ffmpeg.SUPPORTED_MAJOR}.0; it may fail")
    shown: dict[str, int] = {}

    def on_event(event: object) -> None:
        # One line per 10 % per phase: readable in a terminal and in CI logs.
        if isinstance(event, Progress) and event.fraction is not None:
            step = int(event.fraction * 10)
            if shown.get(event.message, -1) < step:
                shown[event.message] = step
                print(f"  {event.message}: {step * 10}%", flush=True)

    cancel = CancelToken()
    try:
        return video.import_video(
            project, path, tool, frames=args.frames, on_event=on_event, cancel=cancel
        )
    except KeyboardInterrupt:
        cancel.cancel()
        raise


def _cmd_upload(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    session = UploadSession(project, port=args.port)
    url = session.start()
    segno.make(url, error="m").terminal(compact=True)
    print(f"scan the code with the phone, or open {url}")
    print("waiting until the phone taps \u201cI'm done\u201d (Ctrl+C to cancel)...", flush=True)
    try:
        shown = -1
        while not session.phone_done:
            time.sleep(0.5)
            received = [f for f in session.status() if f.complete]
            if len(received) != shown:
                shown = len(received)
                print(f"  {shown} file(s) received", flush=True)
    except KeyboardInterrupt:
        session.close()
        print("cancelled; nothing imported", file=sys.stderr)
        return 130
    bundle = session.finish()
    print(f"{len(bundle.files)} files -> capture {bundle.id}")
    _check_photos(project)
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    since = time.time() - args.since * 60 if args.since else None
    watch = FolderWatch(args.folder, since=since, settle=args.settle)
    print(
        f"watching {args.folder}: photos are imported once nothing new has arrived for "
        f"{args.settle:.0f} s (Ctrl+C to stop)",
        flush=True,
    )
    shown = None
    try:
        while True:
            state = watch.poll()
            counts = (len(state.ready), state.arriving, state.videos)
            if counts != shown:
                shown = counts
                print(f"  {state.describe()}", flush=True)
            if state.settled:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopped; nothing imported", file=sys.stderr)
        return 130
    bundle = import_ready(project, state, args.folder)
    print(f"{len(bundle.files)} files -> capture {bundle.id}")
    _check_photos(project)
    return 0


def _cmd_photos(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    for refs, excluded in ((args.exclude, True), (args.include, False)):
        for bundle, names in _resolve_photos(list_bundles(project), refs):
            bundle.set_excluded(names, excluded)
            verb = "left out" if excluded else "brought back"
            print(f"{verb}: {', '.join(f'{bundle.id}/{n}' for n in names)}")
    _check_photos(project)
    return 0


def _resolve_photos(
    bundles: Sequence[CaptureBundle], refs: Sequence[str]
) -> list[tuple[CaptureBundle, list[str]]]:
    """Group `file` or `<capture id>/<file>` references by bundle."""
    found: dict[str, tuple[CaptureBundle, list[str]]] = {}
    for ref in refs:
        capture, _, name = ref.rpartition("/")
        matches = [
            b
            for b in bundles
            if (not capture or b.id == capture) and any(f.name == name for f in b.files)
        ]
        if not matches:
            raise CaptureError(f"no photo {ref!r} in this project")
        if len(matches) > 1:
            ids = ", ".join(b.id for b in matches)
            raise CaptureError(f"{name!r} is in several captures ({ids}); use <capture id>/{name}")
        found.setdefault(matches[0].id, (matches[0], []))[1].append(name)
    return list(found.values())


def _check_photos(project: Project) -> None:
    """Inspect photos not inspected yet, then print the findings."""
    bundles = list_bundles(project)
    for bundle in bundles:
        if photos.needs_inspection(bundle):
            print(f"checking the photos of capture {bundle.id}...", flush=True)
            photos.inspect_bundle(bundle)
    findings = photos.check_project(bundles)
    excluded = [f"{b.id}/{f.name}" for b in bundles for f in b.excluded]
    if excluded:
        print(f"left out ({len(excluded)}): {', '.join(excluded)}")
    if not findings:
        print("photo checks: nothing to report")
        return
    print(f"photo checks: {len(findings)} to look at")
    for finding in findings:
        where = f" [{finding.capture}]" if finding.capture else ""
        print(f"  {finding.level}{where}: {finding.message}")
        if len(finding.files) > 1:
            shown = ", ".join(finding.files[:10])
            more = f" and {len(finding.files) - 10} more" if len(finding.files) > 10 else ""
            print(f"    {shown}{more}")


def _cmd_masks(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    bundles = list_bundles(project)
    if not bundles:
        raise CaptureError("this project has no photos yet; import some first")
    if args.import_folder is not None:
        bundle = _one_capture(bundles, args.capture)
        names = masks.import_folder_masks(project, bundle, args.import_folder)
        print(f"{len(names)} of {len(masks.maskable(bundle))} masks imported into {bundle.id}")
    elif args.drop or args.restore:
        for bundle, names in _resolve_photos(bundles, args.drop or args.restore):
            (masks.drop if args.drop else masks.restore)(project, bundle, names)
            verb = "dropped" if args.drop else "restored"
            print(f"{verb}: {', '.join(f'{bundle.id}/{n}' for n in names)}")
    elif args.clear:
        removed = sum(masks.clear_auto(project, b) for b in bundles)
        print(f"{removed} automatic masks removed")
    elif not args.status:
        _make_masks(project, bundles, force=args.force)
    _print_masks(project, bundles)
    _print_sides(project)
    return 0


def _one_capture(bundles: Sequence[CaptureBundle], capture: str | None) -> CaptureBundle:
    if capture is None:
        if len(bundles) > 1:
            ids = ", ".join(b.id for b in bundles)
            raise CaptureError(f"this project has several captures ({ids}); choose with --capture")
        return bundles[0]
    for bundle in bundles:
        if bundle.id == capture:
            return bundle
    raise CaptureError(f"no capture {capture!r} in this project")


def _make_masks(project: Project, bundles: Sequence[CaptureBundle], *, force: bool) -> None:
    model = masks.find_model()
    if model is None:
        print(f"downloading the masking model ({masks.MODEL.size / 1e6:.0f} MB)...", flush=True)
        shown = [-1]

        def on_progress(received: int, total: int) -> None:
            step = received * 10 // max(total, 1)
            if step > shown[0]:
                shown[0] = step
                print(f"  {step * 10}%", flush=True)

        model = masks.download_model(on_progress=on_progress)
    shown_steps: dict[str, int] = {}

    def on_event(event: object) -> None:
        if isinstance(event, Progress) and event.fraction is not None:
            step = int(event.fraction * 10)
            if shown_steps.get("masks", -1) < step:
                shown_steps["masks"] = step
                print(f"  masking: {step * 10}%", flush=True)

    cancel = CancelToken()
    outcome: list[list[masks.MaskRun] | BaseException] = []

    def work() -> None:
        try:
            outcome.append(
                masks.make_masks(
                    project, bundles, model, on_event=on_event, cancel=cancel, force=force
                )
            )
        except BaseException as exc:  # handed to the main thread
            outcome.append(exc)

    # As for `run`: the worker runs in its own session, so Ctrl+C cancels it here.
    worker = threading.Thread(target=work, name="masks")
    worker.start()
    while worker.is_alive():
        try:
            worker.join(timeout=0.2)
        except KeyboardInterrupt:
            cancel.cancel()
    result = outcome[0]
    if isinstance(result, BaseException):
        raise result
    for run in result:
        if run.reused:
            print(f"capture {run.capture}: masks unchanged")
        else:
            dropped = f", {run.dropped} dropped (nothing found)" if run.dropped else ""
            print(f"capture {run.capture}: {run.added} new masks{dropped}")


def _print_masks(project: Project, bundles: Sequence[CaptureBundle]) -> None:
    entries = masks.review(project, bundles)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.state] = counts.get(entry.state, 0) + 1
    order = ("automatic", "imported", "dropped", "none")
    print("masks: " + ", ".join(f"{counts[s]} {s}" for s in order if s in counts))
    flagged = [e for e in entries if e.flags and e.state != "none"]
    if flagged:
        print(f"to look at ({len(flagged)}):")
        for entry in flagged:
            state = " (dropped)" if entry.state == "dropped" else ""
            print(f"  {entry.capture}/{entry.name}{state}: {', '.join(entry.flags)}")
    if not masks.has_masks(project):
        return
    print("`ez2d run` uses the masks; `--no-masks` ignores them")


def _cmd_crop(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    if args.clear:
        crop.save(project, None)
        print("crop box removed: OpenMVS estimates the region itself")
        return 0
    if args.auto or args.set is not None:
        placement = crop.camera_run(project)
        if placement is None:
            raise PipelineError("place the cameras first (`ez2d run --sparse-only`)")
        if args.auto:
            upright_box = crop.automatic_for(project)
            if upright_box is None:
                raise PipelineError("the camera placement has no sparse points")
        else:
            c, h = args.set[:3], args.set[3:]
            if min(h) <= 0:
                raise PipelineError("half sizes must be positive")
            upright_box = crop.UprightBox((c[0], c[1], c[2]), (h[0], h[1], h[2]), args.yaw)
        crop.save(
            project, crop.from_upright(upright_box, views.upright_rotation(project), placement)
        )
    box = crop.stored(project)
    if box is None:
        print("no crop box: OpenMVS estimates the region from the sparse points")
        return 0
    shown = crop.to_upright(box, views.upright_rotation(project))
    centre = ", ".join(f"{v:.4g}" for v in shown.centre)
    size = ", ".join(f"{2 * v:.4g}" for v in shown.half_size)
    print(f"crop box: centre ({centre}), size ({size}), turned {shown.yaw:.1f}°")
    if crop.current(project) is None:
        print("  drawn on an earlier camera placement: not used until set again")
    return 0


def _cmd_scale(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    if args.clear:
        scale.save(project, None)
        print("scale removed: exports are in the reconstruction's own units")
        return 0
    upright = views.upright_rotation(project)
    if args.points is not None or args.distance is not None:
        if args.distance is None:
            raise PipelineError("give the real distance between the points (--distance MM)")
        placement = crop.camera_run(project)
        if placement is None:
            raise PipelineError("place the cameras first (`ez2d run --sparse-only`)")
        if args.points is not None:
            a, b = args.points[:3], args.points[3:]
            points = scale.from_upright(((a[0], a[1], a[2]), (b[0], b[1], b[2])), upright)
        else:
            picked = scale.current(project)
            if picked is None:
                raise PipelineError("pick two points first (3D view, or --points)")
            points = picked.points
        try:
            scale.save(project, scale.make(points, args.distance, placement))
        except scale.ScaleError as exc:
            raise PipelineError(str(exc)) from exc
    stored = scale.stored(project)
    if stored is None:
        print("no scale: exports are in the reconstruction's own (arbitrary) units")
        return 0
    a, b = scale.to_upright(stored.points, upright)
    shown = " and ".join("(" + ", ".join(f"{v:.4g}" for v in p) + ")" for p in (a, b))
    print(f"scale: {scale.describe(stored)}")
    print(f"  points {shown}")
    if scale.current(project) is None:
        print("  picked on an earlier camera placement: not used until set again")
    return 0


def _cmd_orient(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    if args.auto:
        upright.change(project, None)
        print("orientation: back to the estimate from the photos")
    elif args.level is not None or args.tilt is not None or args.turn is not None:
        try:
            start = upright.starting_point(project)
            if args.level is not None:
                v = args.level
                shown = start.rotation
                points = tuple(
                    upright.mul_transposed(shown, (v[i], v[i + 1], v[i + 2])) for i in (0, 3, 6)
                )
                centre = upright.camera_centre(project)
                if centre is None:
                    raise upright.OrientationError("the camera placement has no cameras")
                start = upright.levelled(start, points, centre)  # type: ignore[arg-type]
            if args.tilt is not None:
                start = upright.tilted(start, args.tilt[-1], -90.0 if "-" in args.tilt else 90.0)
            if args.turn is not None:
                start = upright.turned(start, args.turn)
        except upright.OrientationError as exc:
            raise PipelineError(str(exc)) from exc
        upright.change(project, start)
    manual = upright.current(project)
    rotation = upright.rotation(project)
    if rotation is None:
        print("orientation: the photos don't say which way is up; the reconstruction's own frame")
    else:
        up = upright.mul_transposed(rotation, (0.0, 1.0, 0.0))
        source = f"corrected, turned {manual.turn:g}°" if manual is not None else "from the photos"
        print(f"orientation: {source}; up is ({', '.join(f'{v:.3f}' for v in up)}) in the "
              "reconstruction's coordinates")  # fmt: skip
    if manual is None and upright.stored(project) is not None:
        print("  a correction made on an earlier camera placement is not used")
    return 0


def _cmd_licenses(args: argparse.Namespace) -> int:
    name = licenses.LICENSE if args.gpl else licenses.THIRD_PARTY
    print(licenses.license_text(name) or f"{name} is missing from this copy")
    print(licenses.summary())
    return 0


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    target = args.output or Path.cwd() / diagnostics.default_name(project)
    path = diagnostics.write_diagnostics(project, target)
    print(f"wrote {path}: logs and settings, no photos (the logs contain file paths)")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Exit 0 only if both tools run and are the pinned versions."""
    ok = True
    for name, locate, where, pinned in (
        ("COLMAP", lambda: colmap.locate(args.colmap), "path", colmap.PINNED_VERSION),
        ("OpenMVS", lambda: openmvs.locate(args.openmvs_dir), "bin_dir", openmvs.PINNED_VERSION),
    ):
        try:
            tool = locate()
        except BackendError as exc:
            print(f"{name}: {exc}")
            ok = False
            continue
        bundled = " (bundled)" if _is_bundled(getattr(tool, where)) else ""
        state = "ok" if tool.supported else f"not the tested version {pinned}"
        print(f"{name} {tool.version}: {state}, {getattr(tool, where)}{bundled}")
        ok = ok and tool.supported
    # Only splats need Brush: reported, but not required.
    try:
        splats = brush.locate(args.brush)
    except BackendError as exc:
        print(f"Brush (for splats): {exc}")
    else:
        bundled = " (bundled)" if _is_bundled(splats.path) else ""
        state = "ok" if splats.supported else f"not the tested version {brush.PINNED_VERSION}"
        print(f"Brush {splats.version} (for splats): {state}, {splats.path}{bundled}")
    # Only video import needs FFmpeg: reported, but not required.
    try:
        video_tool = ffmpeg.locate(args.ffmpeg)
    except BackendError as exc:
        print(f"FFmpeg (for video import): {exc}")
    else:
        state = "ok" if video_tool.supported else f"older than {ffmpeg.SUPPORTED_MAJOR}.0"
        print(f"FFmpeg {video_tool.version} (for video import): {state}, {video_tool.path}")
    model = masks.find_model()
    if model is not None:
        print(f"Masking model {masks.MODEL.name}: ok, {model}")
    else:
        print(
            f"Masking model {masks.MODEL.name}: not downloaded yet "
            f"({masks.MODEL.size / 1e6:.0f} MB, the first `ez2d masks` gets it)"
        )
    gpus = hardware.detect_gpus()
    for gpu in gpus:
        note = " (software renderer: too slow for splats)" if gpu.is_cpu else ""
        print(f"GPU: {hardware.describe(gpu)}{note}")
    if not gpus:
        print("GPU: none found (vulkaninfo missing or no Vulkan driver)")
    return 0 if ok else 1


def _is_bundled(path: Path) -> bool:
    bundled = bundled_bin_dir()
    return bundled is not None and path.is_relative_to(bundled)


def _cmd_status(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    print(f"{project.name} ({project.root})")
    bundles = list_bundles(project)
    print(f"captures: {len(bundles)}")
    for bundle in bundles:
        kinds = f"{len(bundle.images)} images, {len(bundle.videos)} videos"
        if bundle.excluded:
            kinds += f" ({len(bundle.excluded)} left out)"
        side = "  (turned over)" if bundle.flipped else ""
        print(f"  {bundle.id}  {bundle.source:<8} {kinds}{side}")
    print("stages:")
    for stage in ALL_STAGES:
        manifest = load_manifest(project.stage_dir(stage))
        if manifest is None:
            state = "-"
        else:
            rss = f", peak {manifest.peak_rss_mb:.0f} MB" if manifest.peak_rss_mb else ""
            state = f"{manifest.status} in {manifest.wall_s:.1f} s{rss} ({manifest.finished})"
        print(f"  {stage:<11} {state}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    tools = Tools(
        colmap=colmap.locate(args.colmap),
        openmvs=openmvs.locate(args.openmvs_dir),
        brush=brush.locate(args.brush) if args.splat else None,
    )
    checked: list[tuple[str, colmap.Colmap | openmvs.OpenMVS | brush.Brush, str]] = [
        ("COLMAP", tools.colmap, colmap.PINNED_VERSION),
        ("OpenMVS", tools.openmvs, openmvs.PINNED_VERSION),
    ]
    if tools.brush is not None:
        checked.append(("Brush", tools.brush, brush.PINNED_VERSION))
    for name, tool, pinned in checked:
        if not tool.supported:
            print(
                f"warning: {name} {tool.version} found, {pinned} is the supported version; "
                f"stages may fail",
                file=sys.stderr,
            )
    quality = args.quality or presets.parse_quality(project.preset)
    if project.preset != quality:
        project.preset = quality
        project.save()
    settings = _settings(args, quality)
    print(f"quality: {presets.LABELS[quality]}")
    for label, value in presets.describe(settings):
        print(f"  {label}: {value}")
    printer = _Printer(sys.stdout, verbose=args.verbose)
    cancel = CancelToken()
    outcome: list[SparseResult | MeshResult | SplatResult | BaseException] = []

    def work() -> None:
        run = run_sparse if args.sparse_only else run_dense if args.dense_only else run_mesh
        try:
            if args.splat:
                outcome.append(
                    run_splat(
                        project,
                        tools,
                        settings,
                        on_event=printer,
                        cancel=cancel,
                        force_from=args.force_from,
                        allow_software_gpu=args.allow_software_gpu,
                    )
                )
                return
            outcome.append(
                run(
                    project,
                    tools,
                    settings,
                    on_event=printer,
                    cancel=cancel,
                    force_from=args.force_from,
                )
            )
        except BaseException as exc:  # handed to the main thread
            outcome.append(exc)

    # The pipeline runs on a worker so Ctrl+C here can cancel it cleanly
    # (backends run in their own session and don't see the terminal's SIGINT).
    worker = threading.Thread(target=work, name="pipeline")
    worker.start()
    while worker.is_alive():
        try:
            worker.join(timeout=0.2)
        except KeyboardInterrupt:
            if not cancel.cancelled:
                printer.line("cancelling...")
                cancel.cancel()
    printer.end_progress()

    result = outcome[0]
    if isinstance(result, PipelineCancelled):
        print("cancelled", file=sys.stderr)
        return 130
    if isinstance(result, StageFailed):
        print(f"error: {result}", file=sys.stderr)
        for line in result.tail:
            print(f"  | {line}", file=sys.stderr)
        print(f"full log: {result.log}", file=sys.stderr)
        return 1
    if isinstance(result, BaseException):
        raise result
    if isinstance(result, SparseResult):
        print(
            f"sparse model: {result.model} "
            f"({result.registered_images}/{result.total_images} images)"
        )
    elif isinstance(result, SplatResult):
        print("splats:")
        for path in result.exports or [result.file]:
            print(f"  {path}")
    else:
        print("textured mesh:")
        for path in result.exports or result.files:
            print(f"  {path}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    if not args.formats:
        print("error: no formats", file=sys.stderr)
        return 2
    files = export_mesh(project, args.formats, align=not args.no_align)
    for path in files:
        print(path)
    for note in export_notes(files):
        print(f"note: {note}")
    return 0


def _formats(text: str) -> tuple[ExportFormat, ...]:
    if text.strip().lower() == "none":
        return ()
    names = [n.strip().lower() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in FORMATS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown format {unknown[0]!r}; use {', '.join(FORMATS)}")
    return tuple(cast(ExportFormat, n) for n in names)


def _settings(args: argparse.Namespace, quality: presets.Quality) -> MeshSettings:
    settings = presets.mesh_settings(
        quality,
        level=args.level,
        refine=args.refine,
        max_image_size=args.max_image_size,
        faces=args.faces,
        steps=args.steps,
    )
    threads = args.threads
    return replace(
        settings,
        features=replace(settings.features, threads=threads),
        mapper=replace(settings.mapper, kind=args.mapper, threads=threads),
        densify=replace(settings.densify, threads=threads),
        mesh=replace(settings.mesh, threads=threads),
        refine=None if settings.refine is None else replace(settings.refine, threads=threads),
        texture=replace(settings.texture, threads=threads),
        export_formats=args.export,
        use_masks=not args.no_masks,
        align=not args.no_align,
    )


class _Printer:
    """Prints pipeline events; progress redraws one line on a terminal."""

    def __init__(self, out: TextIO, *, verbose: bool) -> None:
        self.out = out
        self.verbose = verbose
        self.tty = out.isatty()
        self._progress_shown = False
        self._lock = threading.Lock()

    def __call__(self, event: PipelineEvent) -> None:
        if isinstance(event, StageStarted):
            self.line(f"[{event.index}/{event.count}] {event.stage}")
        elif isinstance(event, StageFinished):
            m = event.manifest
            if event.reused:
                self.line("  unchanged, reused")
            else:
                rss = f", peak {m.peak_rss_mb:.0f} MB" if m.peak_rss_mb else ""
                self.line(f"  {m.status} in {m.wall_s:.1f} s{rss}")
        elif isinstance(event, Notice):
            self.line(f"  note: {event.message}")
        elif isinstance(event, StageOutput):
            inner = event.event
            if isinstance(inner, Output) and self.verbose:
                self.line(f"  | {inner.line}")
            elif isinstance(inner, Progress) and self.tty and not self.verbose:
                percent = f" {inner.fraction:.0%}" if inner.fraction is not None else ""
                with self._lock:
                    self.out.write(f"\r\033[K  {inner.message}{percent}")
                    self.out.flush()
                    self._progress_shown = True

    def line(self, text: str) -> None:
        with self._lock:
            if self._progress_shown:
                self.out.write("\r\033[K")
                self._progress_shown = False
            self.out.write(text + "\n")
            self.out.flush()

    def end_progress(self) -> None:
        with self._lock:
            if self._progress_shown:
                self.out.write("\r\033[K")
                self.out.flush()
                self._progress_shown = False


if __name__ == "__main__":
    raise SystemExit(main())
