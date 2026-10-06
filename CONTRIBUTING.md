# Contributing

Thanks for helping. EZ2DIGITIZE is GPL-3.0-or-later; by contributing you
agree to license your work under it.

## Getting started

```sh
uv sync --all-groups     # Python 3.12 and every dependency group
uv run ez2digitize       # the app (from source it needs the backends: see below)
uv run pytest            # tests; Qt runs offscreen, no GPU or network needed
uv run ruff check        # lint (--fix to autofix)
uv run ruff format       # format
uv run mypy              # strict type check
```

CI runs those four on Linux and macOS; run them before sending a change.
The backends (COLMAP, OpenMVS) are built by `tools/backends/build.sh`, or
take the `backends-Linux` / `backends-macOS` artifact of the Backends
workflow and point `EZ2D_COLMAP` / `EZ2D_OPENMVS_DIR` at it. Tests that need
real backends skip themselves when they're missing.

## How the code is organised

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first; the plan is in
[docs/ROADMAP.md](docs/ROADMAP.md). The rules that matter most (the full
list is in [CLAUDE.md](CLAUDE.md)):

1. Reconstruction tools run as separate processes, through the process
   runner in `ez2digitize.core`, never linked or imported.
2. Only `ez2digitize.ui` and `ez2digitize.app` may import Qt; everything
   else runs headless (CLI, CI).
3. Every pipeline step writes a `stage.json` manifest; caching is decided
   from manifests only.
4. Project format changes bump `schema_version` and add a migration.
5. Photos arrive as capture bundles (originals untouched plus
   `capture.json`); masks are first-class.
6. Backend versions are pinned; a bump is its own change and must pass the
   regression datasets.

## Licenses

- Every source file starts with the SPDX header:
  ```
  # SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```
- A new dependency or backend goes into `THIRD_PARTY_LICENSES` in the same
  change, and must be GPL-3.0-compatible (MIT, BSD, Apache-2.0, LGPL, GPL,
  AGPL, MPL-2.0). Nothing under non-commercial or research-only terms
  (common for 3D Gaussian splatting code and model weights; check the
  weights' license separately) can be bundled or depended on.
- Don't copy code from other projects without checking its license and
  recording it.

## Style

Python 3.12, full type hints (`mypy --strict` clean), ruff (line length
100), `pathlib` for paths, dataclasses for plain data, I/O at the edges.
Subprocesses take an argument list, never `shell=True`. Tests mirror the
package layout and don't need a GPU or network. Commit messages: an
imperative summary line, then a body explaining why.

## Testing on real hardware

The reference machines are an AMD RX 7900 GRE on Arch Linux and an 8 GB M1
Mac. Results from other GPUs (NVIDIA, Intel) and from Windows are welcome:
open an issue with `ez2d check` and, for a failed run, the diagnostics zip.
