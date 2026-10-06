#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Pack the exact source of the backends build.sh builds, for a release:
# the corresponding source the GPL and AGPL ask for (OpenMVS is AGPL-3.0,
# parts of CGAL GPL-3.0+, several dependencies LGPL).
#
#   tools/backends/collect_sources.sh     # Linux x86_64 or macOS arm64
#
# Output: build/backend-sources/ez2d-backends-source-<os>-<arch>.tar with
#   colmap-<tag>/, openmvs-<tag>/   the projects as tagged
#   ez2digitize-backends/           tools/backends from this repository:
#                                   build.sh, pins.sh, patches/ (how they are built)
#   vcpkg-<release>/                vcpkg as pinned
#   vcpkg-versioned-ports/          port scripts of the exact versions used,
#                                   where they differ from that release
#   downloads/                      every port's source archive, as vcpkg
#                                   fetched them for these manifests and triplet
#   SOURCES.txt                     what is where, and how to rebuild
#
# Nothing is compiled: `vcpkg install --only-downloads` resolves the same
# manifests, features and triplet as build.sh and downloads their sources.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
# shellcheck source=pins.sh
. "$REPO/tools/backends/pins.sh"
WORK=${EZ2D_SOURCES_WORK:-$REPO/build/backend-sources}

platform  # TRIPLET, OS, ARCH

log() { printf '\n=== %s\n' "$*"; }

mkdir -p "$WORK"
cd "$WORK"
export VCPKG_ROOT=$WORK/vcpkg
export VCPKG_DOWNLOADS=$WORK/downloads
# Port scripts of versions from other baselines (COLMAP pins its own) land
# in the registries cache: keep it here, to pack it.
export X_VCPKG_REGISTRIES_CACHE=$WORK/registries
export VCPKG_DISABLE_METRICS=1
mkdir -p "$VCPKG_DOWNLOADS" "$X_VCPKG_REGISTRIES_CACHE"

log "vcpkg $VCPKG_VERSION"
fetch_vcpkg vcpkg
prefetch_sources "$VCPKG_DOWNLOADS"
log "COLMAP $COLMAP_VERSION"
fetch colmap "$COLMAP_URL" "$COLMAP_VERSION"
[ "$OS" = windows ] && prepare_colmap colmap
log "OpenMVS $OPENMVS_VERSION"
fetch openmvs "$OPENMVS_URL" "$OPENMVS_VERSION"
prepare_openmvs openmvs "$REPO"

download() {  # download <manifest dir>
  # vcpkg reports success even when a download failed (it only halts that
  # port), so its output is checked as well.
  vcpkg/vcpkg install --only-downloads \
    --triplet "$TRIPLET" --host-triplet "$TRIPLET" \
    --x-manifest-root="$1" --x-install-root="$WORK/$1-installed" \
    "${@:2}" 2>&1 | tee "$WORK/$1-downloads.log"
  if grep -q "Download failed" "$WORK/$1-downloads.log"; then
    echo "error: some sources of $1's dependencies could not be downloaded:" >&2
    grep -B1 "Download failed" "$WORK/$1-downloads.log" >&2 || true
    exit 1
  fi
}
log "sources of COLMAP's dependencies"
download colmap --x-no-default-features   # as build.sh: VCPKG_MANIFEST_NO_DEFAULT_FEATURES
log "sources of OpenMVS's dependencies"
download openmvs

if find downloads -maxdepth 1 -name '*.part' | grep -q .; then
  echo "error: incomplete downloads left in $WORK/downloads" >&2
  exit 1
fi

log "packing"
OUT=$WORK/ez2d-backends-source-$OS-$ARCH
rm -rf "$OUT"
mkdir -p "$OUT"
git -C colmap archive --prefix="colmap-$COLMAP_VERSION/" HEAD | tar -x -C "$OUT"
git -C openmvs archive --prefix="openmvs-$OPENMVS_VERSION/" HEAD | tar -x -C "$OUT"
git -C vcpkg archive --prefix="vcpkg-$VCPKG_VERSION/" HEAD | tar -x -C "$OUT"
mkdir -p "$OUT/ez2digitize-backends"
cp -R "$REPO/tools/backends/build.sh" "$REPO/tools/backends/pins.sh" \
  "$REPO/tools/backends/patches" "$REPO/tools/backends/README.md" "$OUT/ez2digitize-backends/"
if [ -d "$X_VCPKG_REGISTRIES_CACHE/git-trees" ]; then
  cp -R "$X_VCPKG_REGISTRIES_CACHE/git-trees" "$OUT/vcpkg-versioned-ports"
fi
# Port sources only: vcpkg downloads its own tools (cmake, ninja...) to the
# same folder; they are listed in its vcpkg-tools.json.
mkdir -p "$OUT/downloads"
"$PYTHON" - vcpkg/scripts/vcpkg-tools.json downloads "$OUT/downloads" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
tools = {t.get("archive") for t in json.load(open(sys.argv[1]))["tools"]}
for path in sorted(Path(sys.argv[2]).iterdir()):
    if path.is_file() and path.name not in tools and not path.name.endswith(".part"):
        shutil.copy2(path, Path(sys.argv[3]) / path.name)
PYEOF

cat > "$OUT/SOURCES.txt" <<TXT
Source of the EZ2DIGITIZE backends ($OS $ARCH, vcpkg triplet $TRIPLET)

COLMAP   $COLMAP_VERSION   colmap-$COLMAP_VERSION/      ($COLMAP_URL)
OpenMVS  $OPENMVS_VERSION  openmvs-$OPENMVS_VERSION/    ($OPENMVS_URL)
         built with ez2digitize-backends/patches/ applied and the OpenCV
         features set in ez2digitize-backends/pins.sh (prepare_openmvs;
         on Windows, prepare_colmap adds GLEW to COLMAP's manifest)
vcpkg    $VCPKG_VERSION    vcpkg-$VCPKG_VERSION/        ($VCPKG_URL)
         COLMAP's manifest pins its own baseline; the port scripts of those
         versions are in vcpkg-versioned-ports/
Dependencies: downloads/ holds each port's source archive as vcpkg
downloaded it. The license of every library is in the app, under
backends/licenses/.

To rebuild: put ez2digitize-backends/ at tools/backends/ in a checkout of
EZ2DIGITIZE and run tools/backends/build.sh (see its README), or follow its
steps by hand with the sources above.
TXT

(cd "$WORK" && tar -cf "$OUT.tar" "$(basename "$OUT")")
log "wrote $OUT.tar ($(du -h "$OUT.tar" | cut -f1)), $(find "$OUT/downloads" -type f | wc -l) port archives"
