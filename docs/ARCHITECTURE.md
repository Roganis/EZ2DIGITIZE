# Architecture

This describes the intended design. Sections marked *(planned)* are not
implemented yet; update this file when they are.

## Layers

```
ez2digitize/
  app.py        GUI entry point (creates QApplication, main window)
  ui/           PySide6 widgets, viewers, Qt adapters for core objects
  core/         headless logic: project model, stages, manifests, runner
  backends/     (planned) one module per external tool: builds command
                lines, parses progress and errors, detects version
```

`core` and `backends` never import Qt. The UI observes them through Qt
adapters (a `QObject` that forwards runner events as signals). The same core
code is used by a headless CLI *(planned)*, which is how regression datasets
run in CI and on the reference machines.

## Project folder *(planned, Phase 2)*

```
my-scan/
  project.json        schema_version, name, settings, chosen preset
  images/             imported photos or extracted video frames
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

## Stage manifest *(planned, Phase 2)*

Each stage folder has a `stage.json`:

- backend name and exact version
- full command line (argument list)
- parameters after preset resolution
- hashes of inputs (files from earlier stages, images, masks)
- start/end time, exit code, peak memory where available
- GPU driver version (Mesa version on Linux)

A stage is skipped when a successful manifest exists whose inputs and
parameters hash to the same value. Changing a stage invalidates every stage
after it.

## Process runner *(planned, Phase 2)*

- Lives in `core`, built on `subprocess.Popen` with an argument list; no
  shell.
- Each backend process starts in its own process group
  (`start_new_session=True` on POSIX) so cancel can kill the whole tree
  (`SIGTERM`, then `SIGKILL` after a timeout). Windows will need a job object.
- stdout/stderr are read on a background thread, written to the stage's
  `log.txt`, and passed line by line to the backend module's progress parser.
- The runner emits plain Python events (started, log line, progress, finished,
  failed). The UI wraps them in Qt signals; the CLI prints them.
- Only one heavy stage runs at a time: the 8 GB M1 cannot fit two.

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
