# EZ2DIGITIZE

Turn photos or video of small objects into textured meshes and Gaussian
splats, on any GPU vendor. EZ2DIGITIZE is a desktop app that drives
open-source reconstruction tools (COLMAP, OpenMVS, Brush) for you.

**Status:** early development ([roadmap](docs/ROADMAP.md)). The mesh
pipeline runs from a first GUI (`uv run ez2digitize`: create a project,
import a folder of photos, Build mesh) and from the command line (below).

Primary platform: Linux with an AMD GPU. Secondary: macOS on Apple Silicon.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                     # create .venv with all dependencies
uv run ez2digitize          # start the app
uv run pytest               # tests
uv run ruff check           # lint
uv run ruff format          # format
uv run mypy                 # type check
```

The headless pipeline needs COLMAP 4.2.1 and OpenMVS 2.4.0
([`tools/backends`](tools/backends/README.md) builds both; point
`EZ2D_COLMAP` and `EZ2D_OPENMVS_DIR` at them, or pass `--colmap` and
`--openmvs-dir`):

```sh
uv run ez2d new ~/scans/skull
uv run ez2d import ~/scans/skull ~/Pictures/skull   # --masks DIR to add masks
uv run ez2d run ~/scans/skull                       # --help for the options
uv run ez2d status ~/scans/skull
```

See [CLAUDE.md](CLAUDE.md) for project rules and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## License

GPL-3.0-or-later. Third-party components are listed in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
