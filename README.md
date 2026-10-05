# EZ2DIGITIZE

Turn photos or video of small objects into textured meshes and Gaussian
splats, on any GPU vendor. EZ2DIGITIZE is a desktop app that drives
open-source reconstruction tools (COLMAP, OpenMVS, Brush) for you.

**Status:** early development (Phase 0 of the [roadmap](docs/ROADMAP.md)).
Nothing is usable yet.

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

See [CLAUDE.md](CLAUDE.md) for project rules and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## License

GPL-3.0-or-later. Third-party components are listed in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
