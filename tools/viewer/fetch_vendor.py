# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vendor the viewer's pinned JavaScript libraries into the app package.

    uv run python tools/viewer/fetch_vendor.py

Writes src/ez2digitize/ui/viewer_web/vendor/ (committed, so the app runs
from source and packages offline) and vendor/VENDOR.json: the packages,
their npm integrity values and the sha256 of every file written, which a
test checks so the vendored files can't drift from the pins unnoticed.
Run it again after changing a pin below, and update THIRD_PARTY_LICENSES.

Packages come from the npm registry and are checked against npm's SHA-512
integrity values. All are MIT licensed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "src" / "ez2digitize" / "ui" / "viewer_web" / "vendor"
NPM = "https://registry.npmjs.org"


@dataclass(frozen=True)
class Package:
    version: str
    url: str
    integrity: str  # npm's dist.integrity for this exact version
    files: dict[str, str]  # path in the tarball -> path under vendor/


THREE_ADDONS = (
    "controls/OrbitControls.js",
    "loaders/PLYLoader.js",
    "loaders/GLTFLoader.js",
    "utils/BufferGeometryUtils.js",  # imported by GLTFLoader
    "utils/SkeletonUtils.js",  # imported by GLTFLoader
    "postprocessing/Pass.js",  # imported by Spark
)

PACKAGES = {
    "three": Package(
        version="0.186.1",
        url=f"{NPM}/three/-/three-0.186.1.tgz",
        integrity="sha512-blFeqb49wRCSGUGj7gtpfnSGHy2lwDk94RhUmS1c/hTby70kvChbWpkJ4Pm1390L"
        "qzzvTmzgXKHPEafJwCb8jA==",
        files={
            "package/build/three.module.js": "three/three.module.js",
            "package/build/three.core.js": "three/three.core.js",
            **{f"package/examples/jsm/{a}": f"three/addons/{a}" for a in THREE_ADDONS},
            "package/LICENSE": "three/LICENSE",
        },
    ),
    "@sparkjsdev/spark": Package(
        version="2.3.1",
        url=f"{NPM}/@sparkjsdev/spark/-/spark-2.3.1.tgz",
        integrity="sha512-K+SyIbkO/dx1TLLudK8Jp3/noS13z/kdNPo6OK2waxQaGy81HBmKGnNBT/t/K9Ew"
        "3dsIXmWqsJXxGfmAu1T5Jg==",
        files={
            "package/dist/spark.module.js": "spark/spark.module.js",
            "package/LICENSE": "spark/LICENSE",
        },
    ),
}


def fetch(name: str, package: Package) -> dict[str, str]:
    with urllib.request.urlopen(package.url, timeout=120) as response:  # noqa: S310 - fixed URL
        data = response.read()
    digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if digest != package.integrity:
        raise SystemExit(f"{name}: integrity mismatch for {package.url}")
    written = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member_name, target in package.files.items():
            member = tar.extractfile(member_name)
            if member is None:
                raise SystemExit(f"{name}: {member_name} missing from package")
            content = member.read()
            out = VENDOR / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            written[target] = hashlib.sha256(content).hexdigest()
    print(f"{name} {package.version}: ok ({len(written)} files)")
    return written


def main() -> None:
    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    manifest: dict[str, object] = {"packages": {}, "files": {}}
    for name, package in PACKAGES.items():
        files = fetch(name, package)
        manifest["packages"][name] = {  # type: ignore[index]
            "version": package.version,
            "integrity": package.integrity,
        }
        manifest["files"].update(files)  # type: ignore[attr-defined]
    (VENDOR / "VENDOR.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
