# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the pinned JavaScript libraries for the viewer spike into web/vendor.

Packages come from the npm registry and are checked against the registry's
SHA-512 integrity values pinned below. Both are MIT licensed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "web" / "vendor"
NPM = "https://registry.npmjs.org"


@dataclass(frozen=True)
class Package:
    url: str
    integrity: str  # npm's dist.integrity for this exact version
    files: dict[str, str]  # path in the tarball -> path under web/vendor


PACKAGES = {
    "three": Package(
        url=f"{NPM}/three/-/three-0.186.1.tgz",
        integrity="sha512-blFeqb49wRCSGUGj7gtpfnSGHy2lwDk94RhUmS1c/hTby70kvChbWpkJ4Pm1390L"
        "qzzvTmzgXKHPEafJwCb8jA==",
        files={
            "package/build/three.module.js": "three/three.module.js",
            "package/build/three.core.js": "three/three.core.js",
            **{
                f"package/examples/jsm/{addon}": f"three/addons/{addon}"
                for addon in (
                    "controls/OrbitControls.js",
                    "loaders/PLYLoader.js",
                    "postprocessing/Pass.js",  # imported by Spark
                )
            },
            "package/LICENSE": "three/LICENSE",
        },
    ),
    "spark": Package(
        url=f"{NPM}/@sparkjsdev/spark/-/spark-2.3.1.tgz",
        integrity="sha512-K+SyIbkO/dx1TLLudK8Jp3/noS13z/kdNPo6OK2waxQaGy81HBmKGnNBT/t/K9Ew"
        "3dsIXmWqsJXxGfmAu1T5Jg==",
        files={
            "package/dist/spark.module.js": "spark/spark.module.js",
            "package/LICENSE": "spark/LICENSE",
        },
    ),
}


def fetch(name: str, package: Package) -> None:
    with urllib.request.urlopen(package.url, timeout=120) as response:  # noqa: S310 - fixed URL
        data = response.read()
    digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if digest != package.integrity:
        raise SystemExit(f"{name}: integrity mismatch for {package.url}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member_name, target in package.files.items():
            member = tar.extractfile(member_name)
            if member is None:
                raise SystemExit(f"{name}: {member_name} missing from package")
            out = VENDOR / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(member.read())
    print(f"{name}: ok ({len(package.files)} files)")


def main() -> None:
    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    for name, package in PACKAGES.items():
        fetch(name, package)


if __name__ == "__main__":
    main()
