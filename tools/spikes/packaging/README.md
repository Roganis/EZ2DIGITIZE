# Packaging spike

Phase 1 question (c): can the Python/PySide6 app be shipped as a normal
desktop download (AppImage on Linux, `.app` on macOS) at an acceptable size
and startup time, with a backend binary bundled inside?

Two variants are built with PyInstaller (onedir mode):

| Variant | Contents |
|---|---|
| `core` | The real app's main window (`ez2digitize.ui`), no QtWebEngine |
| `viewer` | The [viewer spike](../viewer/README.md): QtWebEngine + three.js + Spark |

Both bundle Brush as `backends/brush_app` and, in `--self-test` mode, run it
from inside the bundle to prove a bundled backend can be found and started.

```sh
uv run python tools/spikes/viewer/fetch_vendor.py
uv run --group packaging python tools/spikes/packaging/build.py --brush "$(which brush_app)"
```

Results land in `build/packaging-spike/` (`report.md`, `report.json`, the
AppImages or zipped `.app`s). Without a display, add `--headless`.

The [Packaging spike workflow](../../../.github/workflows/packaging-spike.yml)
builds on Ubuntu 22.04 and Apple Silicon macOS and uploads the AppImage and
the zipped `.app` as artifacts. A downloaded `.app` is quarantined by
macOS because it isn't signed; to try it on the M1:

```sh
unzip ez2digitize-viewer-macos.zip
xattr -dr com.apple.quarantine ez2digitize-viewer.app
ez2digitize-viewer.app/Contents/MacOS/ez2digitize-viewer --self-test
ez2digitize-viewer.app/Contents/MacOS/ez2digitize-viewer path/to/model.ply
```

## Results (Ubuntu 24.04 container, Python 3.12, PySide6 6.11.2, PyInstaller 6.22.3)

| Variant | Format | Size | Files | Startup (median of 3) |
|---|---|---|---|---|
| core | onedir | 353 MB | 268 | 0.32 s |
| core | AppImage | 144 MB | 1 | 0.77 s |
| viewer | onedir | 749 MB | 3010 | 0.44 s |
| viewer | AppImage | 285 MB | 1 | 2.17 s |

All four self-tests passed: the window (or a QtWebEngine page) came up and
the bundled `brush_app --version` ran. The first CI run (before host
libraries were removed) built and passed the self-tests on both Ubuntu
22.04 and Apple Silicon macOS; on CI the AppImages took 0.4 s (core) and
1.1 s (viewer) to start. Startup is time to window shown and
exit, headless. The AppImage times are pessimistic: without FUSE in the
container, every start extracts the whole image first
(`APPIMAGE_EXTRACT_AND_RUN`); with FUSE the image is mounted instead.

Where the size goes:

- **Brush is 163 MB** (143 MB stripped, about 67 MB compressed). It links
  only against libc, so it bundles cleanly, but it is the largest single
  item. Its release build is what it is; worth asking upstream or building
  with size options later.
- **QtWebEngine adds about 140 MB** to the AppImage (285 vs 144 MB) and 2700
  files to the unpacked bundle (mostly locales and resources, some trimmable).
- **Python + Qt Widgets** is roughly 75 MB compressed, including Qt modules
  the core variant doesn't use (Qt Quick/QML, PDF, GTK theme plugin) that
  PyInstaller's PySide6 hooks pull in through plugins. These can be excluded
  later.

A rough projection for the real app's AppImage: ~75 MB Python/Qt, +140 MB
viewer, +67 MB Brush, + COLMAP and OpenMVS with their libraries (not
measured yet; they come from the backend CI builds). So 300-400 MB.

## Findings for the real packaging (Phase 2 and 6)

- **glibc baseline:** PyInstaller copies system libraries from the build
  machine (pango, libstdc++, libsystemd, libxkbcommon...). Built on Ubuntu
  24.04, the bundle needs glibc ≥ 2.38, so it would not run on Ubuntu 22.04
  or Debian 12. Build on the oldest distro to support: the CI build on
  Ubuntu 22.04 needs glibc ≥ 2.35. PySide6 6.11 wheels need glibc ≥ 2.28
  anyway.
- **Host libraries must not be bundled.** The first CI AppImage, run on
  Arch, printed dozens of `Fontconfig error: ... invalid attribute
  'xsi:nil'` lines: the bundled fontconfig from Ubuntu 22.04 (2.13) can't
  parse Arch's newer `/etc/fonts`. The same class of problem hits the GPU
  driver: host Mesa needs the host's (newer) `libstdc++`, and a bundled
  older copy loaded first makes the driver fail, so rendering falls back to
  software. `build.py` now deletes these libraries from the bundle
  (`HOST_LIBRARIES`: fontconfig, FreeType, HarfBuzz, libstdc++/libgcc_s,
  X11/xkbcommon, D-Bus, systemd, SELinux, and Qt's GTK theme plugin with the
  GTK stack it pulls in). Because the bundle is built on the oldest
  supported distro, the host's copies are always the same version or newer.
  Checked with `LD_DEBUG=libs`: both variants now load these from the
  system, self-tests pass, and the AppImages got 7-8 MB smaller.
- **Bundled backends** live in `sys._MEIPASS/backends` (`_internal/` on
  Linux, `Contents/Frameworks` in a `.app`). The app must call them by
  absolute path from there, never via `PATH`.
- **The AppImage** needs only an `AppRun` script, a `.desktop` file and an
  icon around the PyInstaller onedir; appimagetool 1.9.1 builds it.
- **Python is not the bottleneck** for size or startup: 0.3-0.4 s to a
  window from the unpacked bundle. Packaging pain, if any, will come from the
  native backends and their libraries, which a C++ or Rust GUI would have to
  bundle too.
