# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the license notices of the Rust crates inside Brush's release binary.

The apps bundle Brush's `brush_app` as released, with Brush's LICENSE. The
binary also contains some 700 crates under MIT, Apache-2.0, BSD, Zlib and
similar licenses, which ask for their notices to go with it. This collects
them for the pinned Brush version (ez2digitize.backends.brush.PINNED_VERSION):

    uv run python tools/packaging/brush_notices.py      # needs cargo and the network

- which crates: `cargo tree -p brush-app -e normal,no-proc-macro` from Brush's
  Cargo.lock, for each platform the apps ship (compile-time-only crates
  aren't in the binary, and Brush's own crates are under its LICENSE);
- what of each: the license, copyright and NOTICE files the crate ships
  (identical texts printed once); for a crate without any, the SPDX text
  of each license it names, with its authors from Cargo.toml.

The result, tools/packaging/brush-notices.txt, is committed; the package
builders ship it as backends/licenses/brush/THIRD-PARTY-NOTICES.txt and
refuse one made for another Brush version. Run this again when the Brush
pin changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ez2digitize.backends.brush import PINNED_VERSION  # noqa: E402

BRUSH_URL = "https://github.com/ArthurBrussee/brush"
OUT = REPO / "tools" / "packaging" / "brush-notices.txt"
# The platforms the apps ship Brush for.
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-apple-darwin", "x86_64-pc-windows-msvc")
SPDX_TEXTS = "https://raw.githubusercontent.com/spdx/license-list-data/v3.27.0/text"
NOTICE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|UNLICENSE|COPYRIGHT)", re.IGNORECASE)
HEADER = "Third-party notices for Brush v{version}"


@dataclass
class Crate:
    name: str
    version: str
    license: str
    authors: list[str]
    repository: str
    texts: list[int] = field(default_factory=list)  # indexes into the texts


def tree_crates(lines: Iterable[str]) -> set[tuple[str, str]]:
    """(name, version) of each line of `cargo tree --prefix none -f '{p}'`."""
    found = set()
    for line in lines:
        match = re.match(r"^(\S+) v(\S+)", line.strip())
        if match:
            found.add((match.group(1), match.group(2)))
    return found


def license_ids(expression: str) -> list[str]:
    """The SPDX ids in a license expression (also the old "MIT/Apache-2.0" form)."""
    words = re.split(r"[\s()/]+|\bOR\b|\bAND\b|\bWITH\b", expression)
    return list(dict.fromkeys(w for w in words if w and w not in ("OR", "AND", "WITH")))


def notice_files(folder: Path) -> list[Path]:
    """The license, copyright and NOTICE files a crate ships (also in a LICENSES folder)."""
    files = []
    for entry in sorted(folder.iterdir()):
        if not NOTICE_FILE.match(entry.name):
            continue
        if entry.is_dir():
            files += sorted(p for p in entry.rglob("*") if p.is_file())
        elif entry.is_file():
            files.append(entry)
    return files


def render(version: str, crates: list[Crate], texts: list[str], spdx: dict[str, int]) -> str:
    out = [
        HEADER.format(version=version),
        "=" * len(HEADER.format(version=version)),
        "",
        f"Brush (Apache-2.0, {BRUSH_URL})",
        "is bundled with EZ2DIGITIZE as its release binary, unmodified; its own",
        "license is in LICENSE next to this file. The binary is built from the",
        "Rust crates below (from Brush's Cargo.lock: normal dependencies, without",
        "compile-time-only procedural macros, for",
        f"{', '.join(TARGETS)}).",
        "Each crate's license texts follow, by number; a crate that ships none",
        "gets the standard text of the licenses it names. Written by",
        "tools/packaging/brush_notices.py.",
        "",
        "",
        "Crates",
        "------",
        "",
    ]
    for crate in crates:
        line = f"{crate.name} {crate.version}  {crate.license}"
        if crate.repository:
            line += f"  {crate.repository}"
        out.append(line)
        if not crate.texts and crate.authors:
            out.append(f"    Copyright: {', '.join(crate.authors)}")
        refs = crate.texts or [spdx[i] for i in license_ids(crate.license) if i in spdx]
        out.append(f"    Texts: {', '.join(f'[{n + 1}]' for n in refs)}")
    out += ["", "", "License texts", "-------------", ""]
    standard = {n: i for i, n in spdx.items()}
    for n, text in enumerate(texts):
        title = f"[{n + 1}]" + (f" {standard[n]} (standard text)" if n in standard else "")
        out += [title, "", text.strip(), "", "-" * 72, ""]
    return "\n".join(out)


def collect(source: Path, fetch: bool = True) -> str:
    """The notices for the Brush checkout at `source`."""
    wanted: set[tuple[str, str]] = set()
    for target in TARGETS:
        tree = cargo(
            source, "tree", "-p", "brush-app", "--target", target,
            "-e", "normal,no-proc-macro", "--prefix", "none", "-f", "{p}",
        )  # fmt: skip
        wanted |= tree_crates(tree.splitlines())
    metadata = json.loads(cargo(source, "metadata", "--format-version", "1"))
    crates: list[Crate] = []
    texts: list[str] = []
    by_hash: dict[str, int] = {}
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if (package["name"], package["version"]) not in wanted or package["source"] is None:
            continue  # not in the binary, or Brush's own crates (its LICENSE)
        crate = Crate(
            package["name"],
            package["version"],
            package.get("license") or "(none given)",
            package.get("authors") or [],
            package.get("repository") or "",
        )
        for path in notice_files(Path(package["manifest_path"]).parent):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if key not in by_hash:
                by_hash[key] = len(texts)
                texts.append(text)
            if by_hash[key] not in crate.texts:
                crate.texts.append(by_hash[key])
        crates.append(crate)
    spdx: dict[str, int] = {}
    for license_id in sorted({i for c in crates if not c.texts for i in license_ids(c.license)}):
        if fetch:
            spdx[license_id] = len(texts)
            texts.append(spdx_text(license_id))
    return render(PINNED_VERSION, crates, texts, spdx)


def spdx_text(license_id: str) -> str:
    folder = "exceptions/" if license_id.endswith("-exception") else ""
    url = f"{SPDX_TEXTS}/{folder}{license_id}.txt"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        text: str = response.read().decode("utf-8")
    return text


def cargo(source: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv
        ["cargo", *args, "--locked", "--manifest-path", str(source / "Cargo.toml")],  # noqa: S607
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return done.stdout


def check(path: Path, brush_version: str) -> None:
    """Refuse notices made for another Brush than `brush_version` (`brush_app --version`)."""
    first = path.read_text(encoding="utf-8").splitlines()[0]
    wanted = HEADER.format(version=PINNED_VERSION)
    if first != wanted or not brush_version.endswith(f" {PINNED_VERSION}"):
        raise SystemExit(
            f"{path.name} is {first!r}, Brush is {brush_version!r}, the pin {PINNED_VERSION}: "
            "run tools/packaging/brush_notices.py for the pinned Brush"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, help="a Brush checkout at the pinned tag")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        source = args.source
        if source is None:
            source = Path(scratch) / "brush"
            subprocess.run(  # noqa: S603 - fixed argv
                ["git", "clone", "--quiet", "--depth", "1",  # noqa: S607 - git from PATH
                 "--branch", f"v{PINNED_VERSION}", BRUSH_URL, str(source)],
                check=True,
            )  # fmt: skip
        args.out.write_text(collect(source.resolve()), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
