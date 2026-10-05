# CLAUDE.md

Guidance for Claude Code and other contributors working in this repository.

## Project

EZ2DIGITIZE: a GPL-3.0 desktop app that turns photos or video of small
objects into textured meshes (primary output) and Gaussian splats
(secondary), on non-NVIDIA hardware. Python + PySide6 orchestrator that
drives external reconstruction tools. Plan: `docs/ROADMAP.md`. Design:
`docs/ARCHITECTURE.md`.

Reference machines: AMD RX 7900 GRE on Arch Linux (primary, high end) and an
M1 Mac with 8 GB (secondary, low end). Changes must not assume CUDA.

## Commands

```sh
uv sync                  # install everything (Python 3.12, from .python-version)
uv run ez2digitize       # run the app
uv run ez2d --help       # headless CLI: new, import, run, status
uv run pytest            # tests (Qt runs offscreen, see tests/conftest.py)
uv run ruff check        # lint;  --fix to autofix
uv run ruff format       # format
uv run mypy              # strict type check of src/ and tests/
```

CI (`.github/workflows/ci.yml`) runs lint, format check, mypy and pytest on
Linux and macOS with all dependency groups (`uv sync --all-groups`). Run all
four before committing.

The AppImage is built by `tools/packaging/build.py` (see its README).

Phase 1 spikes (viewer, packaging) live in `tools/spikes/`, each with a
README holding its results; same rules as the benchmark tooling.

Phase 1 benchmark tooling lives in `tools/feasibility/` (package
`ez2d_bench`, CLI `bench.py`, needs `--group feasibility`). It is throwaway
spike code: it may use numpy/Pillow/rembg freely, but nothing in `src/` may
import it. Tests that need real backends (`colmap` on PATH) skip themselves
when the tool is missing.

## Architecture rules

1. **Heavy compute runs out of process.** COLMAP, OpenMVS, Brush and any other
   reconstruction backend are run as CLI subprocesses. Never link them, never
   import their Python bindings for compute. Lightweight, permissively licensed
   libraries (EXIF, image checks, mesh export, reading COLMAP models) may run
   in-process.
2. **Qt stays in the UI.** Only `ez2digitize.ui` and `ez2digitize.app` may
   import PySide6 (enforced by ruff TID251). Everything else must run headless
   so the pipeline works from a CLI and in CI.
3. **Every stage writes a manifest** (`stage.json`): backend name, version and
   build (executable hash), exact command line, parameters, input hashes,
   timing, exit code. Caching and resume are decided from manifests only.
4. **Projects are versioned folders.** `project.json` has a `schema_version`;
   any format change bumps it and adds a migration.
5. **Masks are first-class project data** and are passed to every stage that
   can use them.
6. **All captures arrive as capture bundles:** original files, untouched, plus
   a `capture.json`. Folder import, video frames, phone upload and the future
   Android app all produce this; the pipeline only reads bundles.
7. **Backends are pinned.** Each backend has an exact supported version; a bump
   is its own commit and must pass the regression datasets.

## License rules

- Project license: GPL-3.0-or-later. Every source file starts with:
  ```
  # SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```
- Any new dependency or backend must be added to `THIRD_PARTY_LICENSES` in the
  same commit, in the right section (bundled / external backend / dev-only).
- Only GPL-3.0-compatible licenses may be bundled (MIT, BSD, Apache-2.0,
  LGPL, GPL-3.0, AGPL-3.0, MPL-2.0).
- **Never bundle or add as a dependency** anything under a non-commercial,
  research-only or otherwise non-free license (common for 3DGS/SfM research
  code and model weights: check the weights' license separately from the
  code's). Such tools may only be supported as user-installed plugins.
- Do not copy code from other projects without checking its license and
  recording it in `THIRD_PARTY_LICENSES`.

## Coding conventions

- Python 3.12+, full type hints, `mypy --strict` clean.
- Ruff for lint and formatting (line length 100).
- `pathlib` for paths, never string concatenation.
- Subprocesses: always pass an argument list (never `shell=True`), and put the
  call behind the process runner in `ez2digitize.core` so logging,
  cancellation and manifests are handled in one place.
- Prefer dataclasses for plain data; keep I/O at the edges so logic is
  unit-testable.
- Tests live in `tests/`, mirror the package layout, and must not need a GPU
  or network. Tests that need real backends or GPUs are marked and skipped in
  CI.
- Don't add dependencies for small things the standard library covers.
- Commit messages: imperative summary line, body explaining why.
