# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The release script: versions, the changelog, the release notes."""

import datetime
from pathlib import Path

import pytest
import release
from release import ReleaseError, Version

CHANGELOG = """# Changelog

Intro.

## [Unreleased]

### Added

- Coverage rings.

## [0.1.0] - 2026-09-01

- The first release.

[0.1.0]: https://example.org/v0.1.0
"""


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    init = tmp_path / "__init__.py"
    init.write_text('"""The package."""\n\n__version__ = "0.1.0"\n')
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG)
    return init, changelog


def test_versions() -> None:
    assert Version.parse("1.2.3").tag == "v1.2.3"
    candidate = Version.parse("1.2.0rc1")
    assert candidate.prerelease and candidate.numeric == "1.2.0"
    assert not Version.parse("0.1.0").prerelease
    for bad in ("1.2", "v1.2.3", "1.2.3-rc1", "1.2.3.4", ""):
        with pytest.raises(ReleaseError):
            Version.parse(bad)


def test_sections() -> None:
    found = release.sections(CHANGELOG)
    assert found["Unreleased"] == (None, "### Added\n\n- Coverage rings.")
    assert found["0.1.0"] == ("2026-09-01", "- The first release.")  # link line dropped


def test_prepare(files: tuple[Path, Path]) -> None:
    init, changelog = files
    release.prepare(Version.parse("0.2.0"), datetime.date(2026, 10, 6), init, changelog)
    assert release.current_version(init) == Version("0.2.0")
    found = release.sections(changelog.read_text())
    assert found["Unreleased"] == (None, "")
    assert found["0.2.0"] == ("2026-10-06", "### Added\n\n- Coverage rings.")
    # Twice, or with nothing new: refused.
    with pytest.raises(ReleaseError, match="already has a section"):
        release.prepare(Version.parse("0.2.0"), datetime.date(2026, 10, 6), init, changelog)
    with pytest.raises(ReleaseError, match="Unreleased section is empty"):
        release.prepare(Version.parse("0.3.0"), datetime.date(2026, 10, 6), init, changelog)


def test_check(files: tuple[Path, Path]) -> None:
    init, changelog = files
    assert release.check("v0.1.0", init, changelog) == Version("0.1.0")
    assert release.check(None, init, changelog) == Version("0.1.0")  # a dry run
    with pytest.raises(ReleaseError, match="doesn't match"):
        release.check("v0.2.0", init, changelog)
    init.write_text('__version__ = "0.3.0"\n')
    with pytest.raises(ReleaseError, match="no dated section for 0.3.0"):
        release.check("v0.3.0", init, changelog)


def test_notes(files: tuple[Path, Path]) -> None:
    _init, changelog = files
    text = release.notes(Version.parse("0.1.0"), changelog)
    assert text.startswith("- The first release.\n\n## Downloads")
    assert "`EZ2DIGITIZE-0.1.0-x86_64.AppImage`" in text
    assert "`EZ2DIGITIZE-0.1.0-source.tar.gz`" in text
    # A dry run of a version not in the changelog yet: the Unreleased section.
    assert release.notes(Version.parse("0.2.0"), changelog).startswith("### Added")


def test_command_line(
    files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    init, changelog = files
    monkeypatch.setattr(release, "INIT", init)
    monkeypatch.setattr(release, "CHANGELOG", changelog)
    output = tmp_path / "github-output"
    assert release.main(["check", "--tag", "v0.1.0", "--github-output", str(output)]) == 0
    assert output.read_text() == "version=0.1.0\nprerelease=false\n"
    assert release.main(["check", "--tag", "v9.9.9"]) == 1
    assert "doesn't match" in capsys.readouterr().err


def test_the_repository_is_releasable() -> None:
    version = release.current_version()
    text = release.notes(version, release.CHANGELOG)
    assert "## Downloads" in text and "## Source" in text
    assert "Unreleased" in release.sections(release.CHANGELOG.read_text())
