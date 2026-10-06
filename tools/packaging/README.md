# Packaging

Releases are built by the Release workflow from a version tag
([docs/RELEASING.md](../../docs/RELEASING.md)), which calls the three
package workflows below with backends it builds itself. `release.py` sets
the version and prints the release notes.

`build_appimage.py` builds the Linux AppImage: the app (PyInstaller onedir) with the
pinned COLMAP and OpenMVS from
[`tools/backends`](../backends/README.md) inside.

```sh
tools/backends/build.sh            # or download the backends-Linux CI artifact
uv run --group packaging python tools/packaging/build_appimage.py \
    --backends build/backends/ez2d-backends-linux-x86_64.tar.gz \
    --smoke path/to/photos         # optional: test the result
```

Needs `patchelf`. Output in `build/appimage/`.

One file for both front ends:

```sh
./EZ2DIGITIZE-0.0.1-x86_64.AppImage                 # the GUI
./EZ2DIGITIZE-0.0.1-x86_64.AppImage check           # which tools it uses
./EZ2DIGITIZE-0.0.1-x86_64.AppImage run ~/scans/x   # any `ez2d` command
```

How it is put together (the reasons are in the
[packaging spike](../spikes/packaging/README.md)):

- The backend archive is unpacked into `_internal/backends/`, where the app
  looks for bundled tools before `PATH`.
- The backends need libgomp and libgfortran (and, from GCC 12's
  libgfortran, libquadmath) besides the C/C++ runtime. Those are copied
  from the build machine into `backends/lib`; the binaries get an RPATH of
  `$ORIGIN/../lib` and the libraries `$ORIGIN`, so they load the bundled
  copies.
- Fonts, the C++ runtime, X11/xkb, D-Bus/systemd and the GPU stack are
  removed from the bundle: the host's copies must be used (the spike found
  font errors and a degraded GPU path otherwise).
- Build on the oldest distro to support. The
  [AppImage workflow](../../.github/workflows/appimage.yml) builds on Ubuntu
  22.04 (glibc 2.35), takes the backends from the newest Backends run, and
  runs the smoke test: GUI start, `check` with both tools bundled, and a
  full pipeline run on the synthetic scene to a GLB.

First CI build (AppImage workflow, Ubuntu 22.04): 138 MB, bundling
libgfortran, libquadmath and libgomp; the smoke test found both tools
bundled at the pinned versions and took the synthetic scene to a GLB in
47 s. The same file also ran `check` successfully on Ubuntu 24.04.

Earlier local build (Ubuntu 24.04 container, so not for distribution): 137 MB,
with both bundled tools found at the pinned versions; the synthetic scene
went through all nine steps with masks, to a GLB, in 69 s.

Brush (the pinned release binary, for splats) is bundled next to the other
backends with `--brush DIR`; the workflow does.

Automatic masks (October 2026): ONNX Runtime and NumPy add about 24 MB to
the macOS zip (168 to 192 MB); the AppImage with Brush is 252.5 MB. The
smoke test runs `masks` with the frozen app, which downloads the model
over HTTPS and runs the `mask-worker` command: 50 s for the 32 synthetic
photos on the Linux runner (download included), 226 s on the macOS one.

3D viewer (October 2026): QtWebEngine makes the AppImage 392.5 MB (from
252.5) and the macOS zip 334 MB (from 192), as the packaging spike
predicted. The self-test checks that the viewer's page loads in the bundle.

## macOS app

```sh
uv run --group packaging python tools/packaging/build_macos.py \
    --backends build/backends/ez2d-backends-macos-arm64.tar.gz \
    --brush build/brush --smoke build/synthetic/images
```

PyInstaller builds `EZ2DIGITIZE.app` with the same entry point (the GUI, or
the CLI: `EZ2DIGITIZE.app/Contents/MacOS/EZ2DIGITIZE check`). The backends
go into `Contents/Resources/backends`, not `Contents/Frameworks`: code
signing treats everything in Frameworks as nested code and refuses plain
files there. Their binaries are already ad-hoc signed by the backend build
and find libomp through `@executable_path/../lib`. The whole app is then
signed ad hoc and zipped with `ditto`. The
[macOS app workflow](../../.github/workflows/macos-app.yml) builds it on
Apple Silicon and runs the same smoke test as the AppImage.

It is not notarized (that needs a paid Apple developer account, Phase 6),
so a downloaded copy has to be un-quarantined once:

```sh
xattr -dr com.apple.quarantine EZ2DIGITIZE.app
```

Not done yet: notarization, AppStream metadata, and the viewer's
QtWebEngine (about +140 MB, see the spike).

## Windows app (community-tested)

```sh
uv run --group packaging python tools/packaging/build_windows.py \
    --backends build/backends/ez2d-backends-windows-x86_64.tar.gz \
    --brush build/brush --smoke build/synthetic/images
```

A portable zip, nothing to install: unzip, run `EZ2DIGITIZE.exe`. One
PyInstaller onedir (from a spec the script writes) holds two programs over
the same files: `EZ2DIGITIZE.exe`, the GUI, without a console window, and
`ez2d.exe`, the CLI (`ez2d check`, `ez2d run ...`), a console program, since
a windowed program can't print to the terminal it was started from. The
backends (built by `tools/backends/build.sh` with MSVC: static, plus
`vcomp140.dll`) go into `_internal/backends`, Brush's `brush_app.exe` (which
needs only Windows' own DLLs) next to them. Backends run with
`CREATE_NO_WINDOW`, so the GUI shows no console windows. The icon is drawn
from the SVG by Qt at build time.

The [Windows app workflow](../../.github/workflows/windows-app.yml) builds
it on `windows-latest` and runs the AppImage's smoke test through
`ez2d.exe`; `EZ2DIGITIZE.exe --self-test` has no console, so for it only its
exit code counts (1 if the window, HEIC decoding or the viewer failed).

Not code-signed (Phase 6): SmartScreen warns on the first start ("More
info", then "Run anyway"). No installer yet. Windows has no maintainer
hardware: CI is its only test until community testers report.
