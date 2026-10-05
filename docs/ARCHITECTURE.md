# Architecture

This describes the intended design. Sections marked *(planned)* are not
implemented yet; update this file when they are.

## Layers

```
ez2digitize/
  app.py        GUI entry point (creates QApplication, main window)
  ui/           PySide6 widgets, viewers, Qt adapters for core objects
  core/         headless logic: project model, capture bundles, stages
                and manifests, process runner
  backends/     one module per external tool: finds it, checks its version,
                builds stage command lines, parses progress
```

`core` and `backends` never import Qt. The UI observes them through Qt
adapters (a `QObject` that forwards runner events as signals). The same core
code is used by a headless CLI *(planned)*, which is how regression datasets
run in CI and on the reference machines.

## Project folder (`core/project.py`, `core/capture.py`)

```
my-scan/
  project.json        schema_version, name, preset, settings
  captures/
    20261005-203200/  one capture bundle per import
      capture.json    source, device, and name/size/SHA-256 of every file
      IMG_0001.jpg    original files, copied byte for byte
  masks/              one mask per image (same file stem), optional
  stages/
    01-features/
      stage.json      manifest (see below)
      log.txt
      ...outputs
    02-matching/
    ...
  exports/            user-facing outputs (OBJ, GLB, STL, PLY)
```

- `project.json` changes bump `SCHEMA_VERSION` and add a step to
  `MIGRATIONS`. Opening an older project migrates it and keeps the old file
  as `project.json.v<N>.bak`; a project from a newer app version is refused.
- A bundle is assembled in `captures/.importing-<id>/` and renamed when
  complete, so an interrupted import never appears as a bundle. Files with
  the same name get a numeric suffix; `original_name` keeps the name they
  arrived with. `CaptureBundle.verify()` re-hashes the files.
- JSON files are written atomically (temporary file, fsync, rename).

## Stage manifest (`core/stage.py`)

Each stage folder has a `stage.json`:

- stage name, a run id (new on every run) and status (succeeded, failed,
  cancelled)
- backend name and exact version
- full command line (argument list)
- parameters after preset resolution
- input fingerprints: `sha256:` for a file, `capture:` for a bundle (from
  the hashes already in its `capture.json`), `tree:` for a folder such as
  the masks, `run:<run id>` for an earlier stage
- start/end time, wall and CPU time, exit code, peak memory where available
- host: OS, release, architecture, app version. *(planned)* GPU driver
  version (Mesa version on Linux), with the first GPU backend.

The cache key hashes the stage name, backend, parameters and inputs, but
not the command line, which holds absolute paths (moving a project must
not invalidate it). A stage is skipped when its manifest says it succeeded
with the same key; otherwise its folder is emptied and it runs again in
that folder. Because later stages reference earlier ones by run id,
re-running a stage invalidates every stage after it without hashing its
outputs.

## Process runner (`core/runner.py`)

- Built on `subprocess.Popen` with an argument list; no shell.
- Each backend process starts in its own process group
  (`start_new_session=True`) so cancel kills the whole tree: `SIGTERM`,
  then `SIGKILL` after a grace period. Anything a finished process leaves
  behind holding its output open is killed too. POSIX only for now;
  Windows will need a job object.
- stdout and stderr are merged (universal newlines, so `\r` progress lines
  are split), written to the stage's `log.txt`, and passed line by line to
  the backend module's progress parser.
- The runner emits plain Python events (`Started`, `Output`, `Progress`)
  on the calling thread, in order. The UI runs it on a worker thread and
  wraps the events in Qt signals; the CLI prints them.
- Peak memory: `ru_maxrss` from `wait4`. On Linux a child inherits its
  parent's high-water mark, so when the value isn't above the app's own
  peak the runner uses `VmHWM` samples taken every 0.5 s instead.
- *(planned)* Only one heavy stage runs at a time: the 8 GB M1 cannot fit
  two. That is the pipeline scheduler's job, not the runner's.

## Backend modules (`backends/`)

- `common.find_tool` looks for an executable in this order: an explicit
  path from settings, an `EZ2D_*` environment variable, the bundle's
  `backends/bin`, then `PATH`. A configured path that is wrong is reported,
  never replaced by another copy.
- `locate()` reads the version (`colmap help`, the OpenMVS banner) and
  compares it with the pinned one; other versions are flagged
  `supported=False`. Option names follow the pinned version only.
- Stage builders return a `StageSpec`; the pipeline runs it with
  `run_stage`. Parameters are the option dataclasses, inputs are
  fingerprints of capture bundles, masks and earlier stages.
- COLMAP reads images straight from `captures/` through an image list, named
  `<capture id>/<file>`; masks are `masks/<capture id>/<file>.png`, COLMAP's
  own convention. Intrinsics are grouped per capture bundle by default.
- Matching works on a copy of the features database, so no stage modifies
  another stage's output.
- OpenMVS tools run with their stage folder as working folder. Scene files
  store image paths relative to that folder, and all stage folders are
  siblings, so those paths resolve the same from every stage.
- `tests/backends/test_real_pipeline.py` runs the whole mesh path on the
  synthetic scene; the Backends workflow runs it against the fresh builds.

## Pipeline *(planned)*

Mesh path (Phase 2):

```
import -> checks -> masks -> features -> matching -> mapping -> undistort
       -> crop box (user) -> OpenMVS densify -> mesh -> texture -> export
```

Splat path (Phase 3) branches after `undistort`:

```
undistort -> Brush training -> PLY export
```
