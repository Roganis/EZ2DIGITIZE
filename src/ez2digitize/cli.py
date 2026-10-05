# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless command line: create a project, import captures, run the pipeline.

    ez2d new ~/scans/skull
    ez2d import ~/scans/skull ~/Pictures/skull-photos
    ez2d run ~/scans/skull                  # everything
    ez2d run ~/scans/skull --sparse-only    # stop before densifying
    ez2d status ~/scans/skull
    ez2d export ~/scans/skull --formats glb # again, e.g. in other formats

The same code the GUI uses; this is how regression datasets run on the
reference machines and in CI.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, cast

from ez2digitize.backends import colmap, openmvs
from ez2digitize.backends.common import BackendError
from ez2digitize.core.capture import CaptureError, import_folder, import_masks, list_bundles
from ez2digitize.core.project import Project, ProjectError
from ez2digitize.core.runner import CancelToken, Output, Progress
from ez2digitize.core.stage import load_manifest
from ez2digitize.export import FORMATS, ExportError, ExportFormat, export_mesh
from ez2digitize.pipeline import (
    STAGES,
    MeshResult,
    MeshSettings,
    Notice,
    PipelineCancelled,
    PipelineError,
    PipelineEvent,
    SparseResult,
    StageFailed,
    StageFinished,
    StageOutput,
    StageStarted,
    Tools,
    run_dense,
    run_mesh,
    run_sparse,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except (ProjectError, CaptureError, BackendError, PipelineError, ExportError) as exc:
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

    imp = sub.add_parser("import", help="import folders of photos as capture bundles")
    imp.add_argument("project", type=Path)
    imp.add_argument("folders", type=Path, nargs="+")
    imp.add_argument(
        "--masks",
        type=Path,
        help="folder of masks in COLMAP naming (<image name>.png), for a single folder",
    )
    imp.set_defaults(func=_cmd_import)

    run = sub.add_parser("run", help="reconstruct a textured mesh")
    run.add_argument("project", type=Path)
    part = run.add_mutually_exclusive_group()
    part.add_argument("--sparse-only", action="store_true", help="stop after camera poses")
    part.add_argument("--dense-only", action="store_true", help="only the OpenMVS stages")
    run.add_argument("--max-image-size", type=int, help="COLMAP feature image size")
    run.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    run.add_argument("--level", type=int, help="OpenMVS resolution level (0 = full size)")
    run.add_argument("--refine", action="store_true", help="run RefineMesh (slow)")
    run.add_argument(
        "--export",
        type=_formats,
        default=("obj", "glb"),
        metavar="FORMATS",
        help="formats to export, comma-separated: obj, glb, ply, or none (default obj,glb)",
    )
    run.add_argument("--no-masks", action="store_true", help="ignore the project's masks")
    run.add_argument("--threads", type=int, help="limit CPU threads of every tool")
    run.add_argument("--force-from", choices=STAGES, help="re-run this stage and all after it")
    run.add_argument("--colmap", type=Path, help="COLMAP executable")
    run.add_argument("--openmvs-dir", type=Path, help="folder with the OpenMVS tools")
    run.add_argument("-v", "--verbose", action="store_true", help="print the tools' output")
    run.set_defaults(func=_cmd_run)

    export = sub.add_parser("export", help="export the textured mesh again")
    export.add_argument("project", type=Path)
    export.add_argument("--formats", type=_formats, default=("obj", "glb"), metavar="FORMATS")
    export.set_defaults(func=_cmd_export)

    status = sub.add_parser("status", help="show captures and stage results")
    status.add_argument("project", type=Path)
    status.set_defaults(func=_cmd_status)
    return parser


def _cmd_new(args: argparse.Namespace) -> int:
    project = Project.create(args.project, name=args.name)
    print(f"created project {project.name!r} in {project.root}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    if args.masks is not None and len(args.folders) != 1:
        print("error: --masks needs exactly one folder to import", file=sys.stderr)
        return 2
    project = Project.open(args.project)
    for folder in args.folders:
        bundle, skipped = import_folder(project, folder)
        print(f"{folder}: {len(bundle.files)} files -> capture {bundle.id}")
        for path in skipped:
            print(f"  skipped (not an image or video): {path.name}")
        if args.masks is not None:
            copied = import_masks(project, bundle, args.masks)
            print(f"  {copied} of {len(bundle.images)} masks imported")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    print(f"{project.name} ({project.root})")
    bundles = list_bundles(project)
    print(f"captures: {len(bundles)}")
    for bundle in bundles:
        kinds = f"{len(bundle.images)} images, {len(bundle.videos)} videos"
        print(f"  {bundle.id}  {bundle.source:<8} {kinds}")
    print("stages:")
    for stage in STAGES:
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
    tools = Tools(colmap=colmap.locate(args.colmap), openmvs=openmvs.locate(args.openmvs_dir))
    for name, tool, pinned in (
        ("COLMAP", tools.colmap, colmap.PINNED_VERSION),
        ("OpenMVS", tools.openmvs, openmvs.PINNED_VERSION),
    ):
        if not tool.supported:
            print(
                f"warning: {name} {tool.version} found, {pinned} is the supported version; "
                f"stages may fail",
                file=sys.stderr,
            )
    settings = _settings(args)
    printer = _Printer(sys.stdout, verbose=args.verbose)
    cancel = CancelToken()
    outcome: list[SparseResult | MeshResult | BaseException] = []

    def work() -> None:
        run = run_sparse if args.sparse_only else run_dense if args.dense_only else run_mesh
        try:
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
    for path in export_mesh(project, args.formats):
        print(path)
    return 0


def _formats(text: str) -> tuple[ExportFormat, ...]:
    if text.strip().lower() == "none":
        return ()
    names = [n.strip().lower() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in FORMATS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown format {unknown[0]!r}; use {', '.join(FORMATS)}")
    return tuple(cast(ExportFormat, n) for n in names)


def _settings(args: argparse.Namespace) -> MeshSettings:
    threads = args.threads
    features = colmap.FeatureOptions(threads=threads)
    if args.max_image_size:
        features = colmap.FeatureOptions(max_image_size=args.max_image_size, threads=threads)
    densify = openmvs.DensifyOptions(threads=threads)
    if args.level is not None:
        densify = openmvs.DensifyOptions(resolution_level=args.level, threads=threads)
    return MeshSettings(
        features=features,
        matching=None,
        mapper=colmap.MapperOptions(kind=args.mapper, threads=threads),
        densify=densify,
        mesh=openmvs.MeshOptions(threads=threads),
        refine=openmvs.RefineOptions(threads=threads) if args.refine else None,
        texture=openmvs.TextureOptions(threads=threads),
        export_formats=args.export,
        use_masks=not args.no_masks,
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
