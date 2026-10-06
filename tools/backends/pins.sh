# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The pinned backend sources, shared by build.sh (which builds them) and
# collect_sources.sh (which packs their exact source for a release).
# Sourced, not run.

COLMAP_VERSION=4.2.1
OPENMVS_VERSION=v2.4.0
VCPKG_VERSION=2026.07.29   # used for OpenMVS; COLMAP pins its own vcpkg baseline
COLMAP_URL=https://github.com/colmap/colmap.git
OPENMVS_URL=https://github.com/cdcseacave/openMVS.git
VCPKG_URL=https://github.com/microsoft/vcpkg.git

fetch() {  # fetch <dir> <git url> <tag>
  if [ ! -d "$1" ]; then
    git clone --quiet --depth 1 --branch "$3" --recurse-submodules --shallow-submodules \
      -c advice.detachedHead=false "$2" "$1"
  fi
}

fetch_vcpkg() {  # fetch_vcpkg <dir>: the vcpkg release, with history for baselines
  if [ ! -d "$1" ]; then
    # Full history (without blobs): manifest baselines need older commits.
    git clone --filter=blob:none "$VCPKG_URL" "$1"
  fi
  git -C "$1" fetch --tags --quiet
  git -C "$1" -c advice.detachedHead=false checkout --quiet "$VCPKG_VERSION"
  [ -x "$1/vcpkg" ] || "$1/bootstrap-vcpkg.sh" -disableMetrics
}

prepare_openmvs() {  # prepare_openmvs <checkout> <repo root>: patches and manifest
  # Upstream fixes released after the pinned version (see each patch's header).
  # A checkout kept from an earlier run may already carry them.
  for patch in "$2"/tools/backends/patches/openmvs-*.patch; do
    if git -C "$1" apply --reverse --check "$patch" 2>/dev/null; then
      echo "already applied: $(basename "$patch")"
    else
      git -C "$1" apply "$patch"
    fi
  done
  # OpenMVS asks for vcpkg's "opencv" with its default features, which on Linux
  # include the GTK GUI backend: a large GTK/X11 build (it failed on at-spi2-core)
  # for windows OpenMVS only opens in debug builds. Ask for OpenCV without
  # default features, keeping what OpenMVS uses: calib3d (stereo matching,
  # speckle filter, rectification) and the image formats it reads and writes.
  python3 - "$1/vcpkg.json" <<'PYEOF'
import json, sys
path = sys.argv[1]
manifest = json.load(open(path))
deps = [
    d for d in manifest["dependencies"]
    if not (isinstance(d, dict) and d["name"] in ("opencv", "opencv4"))
]
deps.append({
    "name": "opencv4",
    "default-features": False,
    "features": ["calib3d", "eigen", "jpeg", "jpegxl", "openexr", "png", "tiff"],
})
manifest["dependencies"] = deps
json.dump(manifest, open(path, "w"), indent=2)
PYEOF
}
