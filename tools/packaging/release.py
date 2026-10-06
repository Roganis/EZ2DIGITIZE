# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Versions and release notes for a release (see docs/RELEASING.md).

The version lives in src/ez2digitize/__init__.py (`__version__`, read by
hatch for the package and by the app's about box, diagnostics and package
names); what changed lives in CHANGELOG.md, one `## [X.Y.Z] - YYYY-MM-DD`
section per release under `## [Unreleased]`. A release is the tag
`vX.Y.Z` on main; the Release workflow builds everything from it.

    release.py prepare 0.2.0        # set the version, date the Unreleased section
    release.py check --tag v0.2.0   # what the Release workflow checks first
    release.py notes 0.2.0          # the release's notes (Markdown)
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INIT = REPO / "src" / "ez2digitize" / "__init__.py"
CHANGELOG = REPO / "CHANGELOG.md"
REPOSITORY = "https://github.com/Roganis/EZ2DIGITIZE"

# X.Y.Z, or a pre-release X.Y.ZaN / bN / rcN (PEP 440, as Python packages spell it).
VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?")
VERSION_LINE = re.compile(r'^__version__ = "([^"]*)"$', re.MULTILINE)
UNRELEASED = "## [Unreleased]"
SECTION = re.compile(r"^## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?\s*$", re.MULTILINE)


class ReleaseError(Exception):
    pass


@dataclass(frozen=True)
class Version:
    text: str

    @classmethod
    def parse(cls, text: str) -> Version:
        if VERSION.fullmatch(text) is None:
            raise ReleaseError(f"{text!r} is not a version like 1.2.3 or 1.2.3rc1")
        return cls(text)

    @property
    def prerelease(self) -> bool:
        match = VERSION.fullmatch(self.text)
        return match is not None and match.group(4) is not None

    @property
    def numeric(self) -> str:
        """X.Y.Z without the pre-release part (macOS bundle versions are numbers only)."""
        match = VERSION.fullmatch(self.text)
        assert match is not None
        return ".".join(match.group(i) for i in (1, 2, 3))

    @property
    def tag(self) -> str:
        return f"v{self.text}"


def current_version(init: Path = INIT) -> Version:
    match = VERSION_LINE.search(init.read_text(encoding="utf-8"))
    if match is None:
        raise ReleaseError(f"no __version__ line in {init}")
    return Version.parse(match.group(1))


def sections(changelog: str) -> dict[str, tuple[str | None, str]]:
    """Each `## [name] - date` section's date and body, by name."""
    found: dict[str, tuple[str | None, str]] = {}
    matches = list(SECTION.finditer(changelog))
    for match, following in zip(matches, [*matches[1:], None], strict=True):
        end = following.start() if following else len(changelog)
        body = changelog[match.end() : end]
        # Link definitions at the end belong to the file, not the last section.
        body = re.sub(r"(?m)^\[[^\]]+\]: \S+\s*$", "", body).strip()
        found[match.group(1)] = (match.group(2), body)
    return found


def prepare(version: Version, today: datetime.date, init: Path, changelog: Path) -> None:
    """Set `version` in the package and turn the Unreleased section into its section."""
    text = changelog.read_text(encoding="utf-8")
    if version.text in sections(text):
        raise ReleaseError(f"CHANGELOG.md already has a section for {version.text}")
    unreleased = sections(text).get("Unreleased")
    if unreleased is None or not unreleased[1]:
        raise ReleaseError("CHANGELOG.md's Unreleased section is empty: say what changed first")
    dated = f"{UNRELEASED}\n\n## [{version.text}] - {today.isoformat()}"
    changelog.write_text(text.replace(UNRELEASED, dated, 1), encoding="utf-8")
    source = init.read_text(encoding="utf-8")
    init.write_text(
        VERSION_LINE.sub(f'__version__ = "{version.text}"', source, count=1), encoding="utf-8"
    )


def check(tag: str | None, init: Path, changelog: Path) -> Version:
    """The version to release; with `tag`, it must be the tag of a dated changelog section."""
    version = current_version(init)
    if tag is None:
        return version  # a dry run: any well-formed version
    if tag != version.tag:
        raise ReleaseError(
            f"tag {tag} doesn't match the version in the code ({version.text}, so {version.tag}): "
            "run tools/packaging/release.py prepare first"
        )
    section = sections(changelog.read_text(encoding="utf-8")).get(version.text)
    if section is None or section[0] is None or not section[1]:
        raise ReleaseError(f"CHANGELOG.md has no dated section for {version.text}")
    return version


def notes(version: Version, changelog: Path) -> str:
    """The release page's text: what changed, then what to download."""
    found = sections(changelog.read_text(encoding="utf-8"))
    _date, changes = found.get(version.text) or found.get("Unreleased") or (None, "")
    v = version.text
    return f"""{changes or "(No changes listed.)"}

## Downloads

- **Linux x86_64:** `EZ2DIGITIZE-{v}-x86_64.AppImage`; make it executable
  and run it.
- **macOS (Apple Silicon):** `EZ2DIGITIZE-{v}-macos-arm64.zip`. Not
  notarized yet: once, run `xattr -dr com.apple.quarantine EZ2DIGITIZE.app`.
- **Windows x64 (community-tested):** `EZ2DIGITIZE-{v}-windows-x86_64.zip`;
  unzip and run `EZ2DIGITIZE.exe`. Not signed yet, so SmartScreen warns once.

Each includes COLMAP, OpenMVS and Brush at the tested versions; see
[QUICKSTART]({REPOSITORY}/blob/{version.tag}/docs/QUICKSTART.md).
`SHA256SUMS` lists every file's checksum.

## Source

EZ2DIGITIZE is GPL-3.0-or-later: `EZ2DIGITIZE-{v}-source.tar.gz` is this
release's source. The bundled COLMAP (BSD) and OpenMVS (AGPL-3.0, with GPL
parts of CGAL) were built from the source in
`ez2d-backends-source-<platform>.tar` (the tagged code, our patches, the
pinned vcpkg and every dependency's source archive), as THIRD_PARTY_LICENSES
offers.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="set the version and date its changelog section")
    p.add_argument("version")
    p.add_argument("--date", type=datetime.date.fromisoformat, default=datetime.date.today())
    c = commands.add_parser("check", help="check the version (and the tag) before a release")
    c.add_argument("--tag", help="the pushed tag, e.g. v0.2.0; without it, a dry run")
    c.add_argument("--github-output", type=Path, help="append version= and prerelease= here")
    n = commands.add_parser("notes", help="print the release notes")
    n.add_argument("version")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            version = Version.parse(args.version)
            prepare(version, args.date, INIT, CHANGELOG)
            print(f"version {version.text}; commit, then tag {version.tag} on main")
        elif args.command == "check":
            version = check(args.tag, INIT, CHANGELOG)
            print(version.text)
            if args.github_output:
                with args.github_output.open("a", encoding="utf-8") as out:
                    out.write(f"version={version.text}\n")
                    out.write(f"prerelease={str(version.prerelease).lower()}\n")
        else:
            print(notes(Version.parse(args.version), CHANGELOG), end="")
    except ReleaseError as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
