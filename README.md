# EZ2DIGITIZE

Turn photos or video of small objects into textured meshes and Gaussian
splats, on any GPU vendor. EZ2DIGITIZE is a desktop app that drives
open-source reconstruction tools (COLMAP, OpenMVS, Brush) for you.

**Status:** early development ([roadmap](docs/ROADMAP.md)). Photos, a
video or a phone upload in; a textured mesh (OBJ, GLB, STL, 3MF) or
Gaussian splats out; from the GUI or the command line. CI builds a Linux
AppImage, a macOS app and a portable Windows zip with the tools inside;
tagged versions are published on the
[releases page](https://github.com/Roganis/EZ2DIGITIZE/releases)
([changelog](CHANGELOG.md)).

Primary platform: Linux with an AMD GPU. Secondary: macOS on Apple Silicon.

- [Quick start](docs/QUICKSTART.md)
- [Capturing a small object](docs/CAPTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing](CONTRIBUTING.md)

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
uv run ez2d import ~/scans/skull ~/Videos/skull.mp4  # a video: --frames N (100)
uv run ez2d upload ~/scans/skull                    # photos from a phone (QR code)
uv run ez2d watch ~/scans/skull ~/Sync/Camera       # or from the folder it syncs to
uv run ez2d photos ~/scans/skull --exclude Preview.jpg  # photo checks, leave out
uv run ez2d masks ~/scans/skull                     # automatic masks (model: 179 MB)
uv run ez2d import ~/scans/skull ~/Pictures/under --flipped  # the other side
uv run ez2d run ~/scans/skull --sparse-only         # place the cameras only, then:
uv run ez2d crop ~/scans/skull --auto               # a crop box (or --set, in the 3D view)
uv run ez2d scale ~/scans/skull --distance 42       # the points picked in the 3D view: 42 mm
uv run ez2d orient ~/scans/skull --tilt x           # lying on its side: a quarter turn
uv run ez2d run ~/scans/skull --quality fast        # fast, balanced (default), high
uv run ez2d status ~/scans/skull
uv run ez2d export ~/scans/skull --formats glb      # OBJ and GLB are exported after run
```

See [CLAUDE.md](CLAUDE.md) for project rules and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## License

GPL-3.0-or-later. Third-party components are listed in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
