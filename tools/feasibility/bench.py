# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 1 feasibility benchmarks. See tools/feasibility/README.md.

uv run --group feasibility tools/feasibility/bench.py <command> --help
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from ez2d_bench import frames, masks, report, synthetic, sysinfo
from ez2d_bench.execute import Executor
from ez2d_bench.plan import PlanError, load
from ez2d_bench.record import RunFailed
from ez2d_bench.tools import Toolbox

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path("~/ez2d-feasibility").expanduser()
DEFAULT_REPORTS = REPO_ROOT / "docs" / "feasibility" / "results"


def _toolbox(args: argparse.Namespace) -> Toolbox:
    return Toolbox.discover(colmap=args.colmap, openmvs_dir=args.openmvs_dir, brush=args.brush)


def cmd_sysinfo(args: argparse.Namespace) -> int:
    print(json.dumps(sysinfo.collect(_toolbox(args)), indent=2))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    n = synthetic.generate(args.out, views_per_ring=args.views_per_ring)
    print(f"wrote {n} images to {args.out / 'images'} and masks to {args.out / 'masks'}")
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    info = frames.extract(args.video, args.out, count=args.count, window=args.window)
    print(json.dumps(info, indent=2))
    return 0


def cmd_masks(args: argparse.Namespace) -> int:
    info = masks.make_masks(args.images, args.out, model=args.model, dilate_px=args.dilate)
    print(json.dumps(info, indent=2))
    print(f"review the preview: {args.out.parent / (args.out.name + '_preview.jpg')}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        plan = load(args.plan)
    except PlanError as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2
    executor = Executor(
        plan,
        results=args.results,
        machine=args.machine,
        toolbox=_toolbox(args),
        only=set(args.only) if args.only else None,
        force=args.force,
    )
    try:
        records = executor.execute()
    except RunFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to continue where it stopped")
        return 130
    failed = [r for r in records if not r.ok]
    print(f"{len(records) - len(failed)} ok, {len(failed)} failed or skipped")
    for r in failed:
        print(f"  {r.kind}/{r.label}: {r.status}: {r.reason}")
    print(f"results in {executor.dataset_dir}")
    print("next: bench.py report")
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    if not args.results.is_dir():
        print(f"no results in {args.results}", file=sys.stderr)
        return 2
    for path in report.write_reports(args.results, args.out):
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    tools = argparse.ArgumentParser(add_help=False)
    tools.add_argument("--colmap", type=Path, help="colmap executable (default: PATH)")
    tools.add_argument(
        "--openmvs-dir",
        type=Path,
        help="folder with the OpenMVS binaries (default: PATH, /usr/local/bin/OpenMVS)",
    )
    tools.add_argument("--brush", type=Path, help="Brush executable (default: brush_app on PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sysinfo", parents=[tools], help="show machine info and tool versions")
    p.set_defaults(func=cmd_sysinfo)

    p = sub.add_parser("synth", help="render a small synthetic dataset for smoke tests")
    p.add_argument("out", type=Path)
    p.add_argument("--views-per-ring", type=int, default=16)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("frames", help="extract the sharpest frames from a video")
    p.add_argument("video", type=Path)
    p.add_argument("out", type=Path, help="output folder for the frames")
    p.add_argument("--count", type=int, default=120, help="number of frames to keep")
    p.add_argument(
        "--window", type=int, default=3, help="candidates per kept frame; the sharpest one is kept"
    )
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("masks", help="background-removal masks in COLMAP naming")
    p.add_argument("images", type=Path)
    p.add_argument("out", type=Path, help="output folder for the masks")
    p.add_argument(
        "--model",
        default="birefnet-general",
        help="rembg model: birefnet-general, birefnet-general-lite, isnet-general-use, ...",
    )
    p.add_argument("--dilate", type=int, default=4, help="grow masks by this many pixels")
    p.set_defaults(func=cmd_masks)

    p = sub.add_parser("run", parents=[tools], help="run the variants of a plan file")
    p.add_argument("plan", type=Path)
    p.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"where run outputs go (large!), default {DEFAULT_RESULTS}",
    )
    p.add_argument(
        "--machine",
        default=socket.gethostname().split(".")[0],
        help="machine name used in the results path (default: hostname)",
    )
    p.add_argument("--only", nargs="+", metavar="LABEL", help="run only these labels")
    p.add_argument("--force", action="store_true", help="rerun runs that already succeeded")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="write Markdown/JSON summaries for docs/feasibility")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Tools run in other working directories (run folders, temp dirs), so make
    # every path from the command line absolute first.
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
