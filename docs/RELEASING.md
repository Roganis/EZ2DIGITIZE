# Releasing

A release is a version tag on main. The Release workflow builds everything
from that tagged commit and makes a **draft** GitHub release; a maintainer
checks it and publishes it.

## What a release contains

| File | What |
|---|---|
| `EZ2DIGITIZE-<v>-x86_64.AppImage` | Linux x86_64, glibc 2.35 or newer |
| `EZ2DIGITIZE-<v>-macos-arm64.zip` | macOS, Apple Silicon; signed ad hoc, not notarized |
| `EZ2DIGITIZE-<v>-windows-x86_64.zip` | Windows x64, portable; not code-signed |
| `EZ2DIGITIZE-<v>-source.tar.gz` | the app's source at the tag (`git archive`) |
| `ez2d-backends-source-<platform>.tar` | the exact source of the bundled COLMAP and OpenMVS, one per platform |
| `SHA256SUMS` | checksums of all of the above |

The backend source archives are the corresponding source the GPL and AGPL
ask for: THIRD_PARTY_LICENSES offers them with each release, so a release
without them is incomplete and the workflow refuses to make one. They come
from `tools/backends/collect_sources.sh` run in the same workflow, from the
same pins (`tools/backends/pins.sh`) the binaries are built from.

## Versions

`__version__` in `src/ez2digitize/__init__.py` is the version: the Python
package, the about box, diagnostics, the package file names and the macOS
bundle all take it from there. Releases use [semantic
versioning](https://semver.org) spelled as Python versions: `0.1.0`, and
pre-releases `0.2.0rc1` (tag `v0.2.0rc1`, marked as a pre-release on GitHub).
Until 1.0, a minor version may change project files or settings (with a
migration, see CLAUDE.md).

## Changelog

`CHANGELOG.md` has a `## [Unreleased]` section at the top. A change people
will notice adds a line there in the same commit. A release turns that
section into `## [<v>] - <date>`; its text becomes the release notes, above
the standard download and source sections (`tools/packaging/release.py
notes <v>` prints them).

## Making a release

1. Be sure main is ready: CI, AppImage, macOS app and Windows app green; the
   Backends workflow green on main (its vcpkg cache is what the release's
   backend build starts from; from a cold cache that build takes hours).
   Check THIRD_PARTY_LICENSES against what the packages contain (anything
   added since the last release?). After a Brush pin bump, regenerate the
   notices of the crates inside it (`tools/packaging/brush_notices.py`):
   the package builds refuse notices made for another Brush.
   After a backend bump, and before a release that changes how the app
   drives the backends, run the Phase 1 plans on the reference machine and
   `bench.py regress check` them against the saved reference (see
   tools/feasibility/README.md): it must pass.
2. On a branch, set the version and date the changelog:

   ```sh
   uv run python tools/packaging/release.py prepare 0.1.0
   ```

   Read the changelog section over, commit ("Release 0.1.0"), and merge to
   main through a pull request.
3. Tag the merged commit on main and push the tag:

   ```sh
   git switch main && git pull
   git tag -a v0.1.0 -m "EZ2DIGITIZE 0.1.0"
   git push origin v0.1.0
   ```

4. The Release workflow (Actions → Release) checks that the tag matches the
   code's version and has a dated changelog section, and that it is on
   main; runs CI; builds the backends from the tagged commit, then the
   AppImage, the macOS app and the Windows app with them, each tested on
   the synthetic scene; packs the backends' source; and makes the draft
   release with every file and `SHA256SUMS`. If any step fails, nothing is
   released: fix it on main, delete the tag (`git push origin :v0.1.0`,
   `git tag -d v0.1.0`) and tag again, or release the next patch version.
5. On the releases page, check the draft: the files are all there, the notes
   read well. Try the AppImage on the reference machine, and the macOS app
   on the M1. Then **Publish release**.

## A dry run

Actions → Release → Run workflow, on any branch, runs everything above
except making the release: the summary lists the files it would attach
with their checksums, and the notes. Worth doing before the first release
of a version that changes the packaging.

## Not yet

- macOS notarization (needs a paid Apple developer account) and Windows
  code signing: until then, users un-quarantine the app on macOS and click
  past SmartScreen on Windows (QUICKSTART says how).
- A Windows installer: the zip is portable.
