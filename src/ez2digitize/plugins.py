# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugins: reconstruction backends the user installs, with their licenses shown.

A plugin replaces one step of the pipeline with a program of its own,
for tools that can't be bundled: research code and model weights under
non-commercial licenses (VGGT, MASt3R-SfM, many 3DGS trainers), or anything
the user prefers. It runs like the bundled backends, as a subprocess through
the process runner, with a stage manifest, caching and cancellation; nothing
of it is imported. See docs/PLUGINS.md for writing one.

A plugin is a folder with `ez2d-plugin.toml`, its license files and whatever
it runs. It fills one slot:

- `poses`, camera placement: instead of COLMAP's features, matching and
  mapping, it writes a COLMAP model of the photos (`sparse/0/cameras.bin`,
  `images.bin`, `points3D.bin`); COLMAP then undistorts as usual, so the
  dense cloud, mesh and splats work unchanged.
- `splats`, splat training: instead of Brush, it trains on the same
  COLMAP dataset Brush gets and writes `splat.ply`.

Installed plugins live in `plugins_dir()`, one folder each. A plugin can be
chosen for its slot only once its licenses were shown and accepted; the
acceptance is for those exact texts, so an update that changes them asks
again. Choices and acceptances are in `plugins.json` there, shared by the
app and `ez2d`.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import string
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast, get_args

from ez2digitize.backends import brush, colmap
from ez2digitize.backends.common import BackendError, executable, find_tool
from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.files import (
    FormatError,
    fingerprint,
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from ez2digitize.core.project import Project
from ez2digitize.core.resources import cpu_threads
from ez2digitize.core.runner import Progress
from ez2digitize.core.stage import (
    Backend,
    StageManifest,
    StageSpec,
    capture_input,
    stage_input,
    tree_input,
)

MANIFEST = "ez2d-plugin.toml"
STATE_FILE = "plugins.json"
API_VERSION = 1  # of the manifest and the slots' contracts
ENV_VAR = "EZ2D_PLUGINS_DIR"

Slot = Literal["poses", "splats"]
SLOTS: tuple[Slot, ...] = get_args(Slot)
SLOT_LABELS: dict[Slot, str] = {"poses": "Camera placement", "splats": "Splats"}
BUILT_IN: dict[Slot, str] = {"poses": "COLMAP", "splats": "Brush"}

# What each slot's command may use, besides {plugin}, {output}, {threads} and
# {colmap} (the COLMAP the app uses: bundled, or set in its settings).
PLACEHOLDERS: dict[Slot, frozenset[str]] = {
    "poses": frozenset({"captures", "image_list", "masks"}),
    "splats": frozenset({"dataset", "steps", "max_resolution", "max_splats", "sh_degree"}),
}
COMMON_PLACEHOLDERS = frozenset({"plugin", "output", "threads", "colmap"})
IMAGE_LIST = "image_list.txt"
MASKS_DIR = "masks"
PLATFORMS = {"linux": "linux", "macos": "darwin", "windows": "win32"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# SPDX identifiers of free licenses (FSF free or OSI approved) that plugins
# commonly use, code and model weights alike. Anything else is shown as "not
# known to be free": often non-commercial or research-only.
FREE_LICENSES = frozenset(
    {
        "0BSD", "AFL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later", "Apache-2.0", "Artistic-2.0",
        "BSD-1-Clause", "BSD-2-Clause", "BSD-3-Clause", "BSL-1.0", "CC-BY-4.0", "CC-BY-SA-4.0",
        "CC0-1.0", "EPL-2.0", "EUPL-1.2", "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only",
        "GPL-3.0-or-later", "ISC", "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only",
        "LGPL-3.0-or-later", "MIT", "MIT-0", "MPL-2.0", "OFL-1.1", "PSF-2.0", "Python-2.0",
        "Unlicense", "UPL-1.0", "Zlib",
    }
)  # fmt: skip


class PluginError(BackendError):
    """A plugin is malformed, missing, not accepted, or can't run here."""


# --- the manifest ------------------------------------------------------------------


@dataclass(frozen=True)
class PluginLicense:
    covers: str  # what it applies to: "code", "model weights"...
    spdx: str  # SPDX expression, or LicenseRef-... for anything else
    file: Path

    @property
    def free(self) -> bool:
        return license_is_free(self.spdx)

    def text(self) -> str:
        return self.file.read_text(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Plugin:
    id: str
    name: str
    version: str
    slot: Slot
    folder: Path
    command: tuple[str, ...]
    licenses: tuple[PluginLicense, ...]
    description: str = ""
    homepage: str = ""
    # Appended to the command when the project has masks (poses only).
    with_masks: tuple[str, ...] = ()
    # Regex for progress lines: named groups `done` and `total`, else the
    # first two groups; `message` labels it.
    progress: str | None = None
    progress_message: str = ""
    gpu: bool = False
    platforms: tuple[str, ...] = ()  # sys.platform values; empty: all
    pty: bool = False

    @property
    def manifest(self) -> Path:
        return self.folder / MANIFEST

    @property
    def backend(self) -> Backend:
        return Backend(f"plugin:{self.id}", self.version)

    @property
    def free(self) -> bool:
        return all(lic.free for lic in self.licenses)

    @property
    def runs_here(self) -> bool:
        return not self.platforms or sys.platform in self.platforms

    def license_digest(self) -> str:
        """Of the license texts: accepting a plugin accepts exactly these."""
        return fingerprint([(lic.covers, lic.spdx, sha256_file(lic.file)) for lic in self.licenses])

    def license_summary(self) -> str:
        return ", ".join(f"{lic.spdx} ({lic.covers})" for lic in self.licenses)

    def label(self) -> str:
        return f"{self.name} {self.version}"


def license_is_free(expression: str) -> bool:
    """Whether an SPDX expression grants a free license (any OR branch, every AND part)."""
    text = expression.replace("(", " ").replace(")", " ")
    choices = re.split(r"\s+OR\s+", text.strip())
    for choice in choices:
        parts = re.split(r"\s+AND\s+", choice.strip())
        ids = [re.split(r"\s+WITH\s+", part.strip())[0].strip().removesuffix("+") for part in parts]
        if ids and all(i in FREE_LICENSES for i in ids):
            return True
    return False


def read_plugin(folder: Path) -> Plugin:
    """The plugin in `folder`, checked against the manifest's rules."""
    path = folder / MANIFEST
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PluginError(f"{folder}: no {MANIFEST}, not a plugin") from None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PluginError(f"cannot read {path}: {exc}") from exc
    where = f"{path}"

    def text(key: str, *, required: bool = True) -> str:
        value = data.get(key, "")
        if not isinstance(value, str) or (required and not value.strip()):
            raise PluginError(f"{where}: '{key}' must be a non-empty string")
        return value.strip()

    def strings(key: str) -> tuple[str, ...]:
        value = data.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise PluginError(f"{where}: '{key}' must be a list of strings")
        return tuple(cast(list[str], value))

    api = data.get("api")
    if api != API_VERSION:
        raise PluginError(
            f"{where}: written for plugin API {api!r}; this version of EZ2DIGITIZE "
            f"supports {API_VERSION}"
        )
    plugin_id = text("id")
    if not ID_PATTERN.match(plugin_id):
        raise PluginError(
            f"{where}: id {plugin_id!r} must be lower case letters, digits, '.', '_' or '-'"
        )
    slot = text("provides")
    if slot not in SLOTS:
        raise PluginError(f"{where}: 'provides' must be one of {', '.join(SLOTS)}")
    command = strings("command")
    if not command:
        raise PluginError(f"{where}: 'command' must name the program to run")
    with_masks = strings("with_masks")
    if with_masks and slot != "poses":
        raise PluginError(f"{where}: 'with_masks' is only for camera placement plugins")
    allowed = COMMON_PLACEHOLDERS | PLACEHOLDERS[slot]
    for arg in (*command, *with_masks):
        for name in _placeholders(arg, where):
            if name not in allowed:
                raise PluginError(
                    f"{where}: unknown placeholder {{{name}}} in {arg!r}; "
                    f"a {slot} plugin can use {', '.join(sorted(allowed))}"
                )
    if any("{masks}" in arg for arg in command):
        raise PluginError(f"{where}: {{masks}} goes in 'with_masks', used only when masked")
    platforms = []
    for name in strings("platforms"):
        if name not in PLATFORMS:
            raise PluginError(f"{where}: platform {name!r}; use {', '.join(PLATFORMS)}")
        platforms.append(PLATFORMS[name])
    licenses = []
    entries = data.get("license", [])
    if not isinstance(entries, list) or not entries:
        raise PluginError(
            f"{where}: every plugin needs at least one [[license]] (covers, spdx, file); "
            "list the model weights' license separately from the code's"
        )
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) and entry[k].strip() for k in ("covers", "spdx", "file")
        ):
            raise PluginError(f"{where}: each [[license]] needs covers, spdx and file")
        file = _inside(folder, entry["file"], where)
        if not file.is_file():
            raise PluginError(f"{where}: license file {entry['file']} is missing")
        licenses.append(PluginLicense(entry["covers"].strip(), entry["spdx"].strip(), file))
    progress = data.get("progress")
    progress_message = ""
    if progress is not None:
        if not isinstance(progress, dict) or not isinstance(progress.get("pattern"), str):
            raise PluginError(f"{where}: [progress] needs a 'pattern' (a regular expression)")
        try:
            compiled = re.compile(progress["pattern"])
        except re.error as exc:
            raise PluginError(f"{where}: progress pattern: {exc}") from exc
        if compiled.groups < 2:
            raise PluginError(f"{where}: the progress pattern needs two groups, done and total")
        progress_message = str(progress.get("message", "") or "")
        progress = progress["pattern"]
    for flag in ("gpu", "pty"):
        if not isinstance(data.get(flag, False), bool):
            raise PluginError(f"{where}: '{flag}' must be true or false")
    return Plugin(
        id=plugin_id,
        name=text("name"),
        version=text("version"),
        slot=slot,
        folder=folder.absolute(),
        command=command,
        licenses=tuple(licenses),
        description=text("description", required=False),
        homepage=text("homepage", required=False),
        with_masks=with_masks,
        progress=progress,
        progress_message=progress_message,
        gpu=bool(data.get("gpu", False)),
        platforms=tuple(platforms),
        pty=bool(data.get("pty", False)),
    )


def _placeholders(arg: str, where: str) -> list[str]:
    try:
        return [name for _text, name, _spec, _conv in string.Formatter().parse(arg) if name]
    except ValueError as exc:
        raise PluginError(f"{where}: {arg!r}: {exc} (write {{{{ for a brace)") from exc


def _inside(folder: Path, relative: str, where: str) -> Path:
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise PluginError(f"{where}: {relative} is outside the plugin's folder")
    return path


# --- installed plugins ------------------------------------------------------------


def plugins_dir() -> Path:
    """Where plugins are installed: the user's data folder, or `EZ2D_PLUGINS_DIR`."""
    if override := os.environ.get(ENV_VAR):
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ez2digitize" / "plugins"


@dataclass(frozen=True)
class Installed:
    plugins: tuple[Plugin, ...]
    # Folders in plugins_dir() that aren't valid plugins, with why.
    problems: tuple[str, ...] = ()

    def get(self, plugin_id: str) -> Plugin | None:
        return next((p for p in self.plugins if p.id == plugin_id), None)

    def for_slot(self, slot: Slot) -> list[Plugin]:
        return [p for p in self.plugins if p.slot == slot]


def installed() -> Installed:
    folder = plugins_dir()
    plugins, problems = [], []
    try:
        children = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return Installed(())
    for child in children:
        try:
            plugin = read_plugin(child)
        except PluginError as exc:
            problems.append(str(exc))
            continue
        if plugin.id != child.name:
            problems.append(f"{child}: holds plugin {plugin.id!r}; its folder must be named so")
            continue
        plugins.append(plugin)
    return Installed(tuple(plugins), tuple(problems))


def install(source: Path, *, replace_existing: bool = False) -> Plugin:
    """Copy a plugin (a folder, or a .zip of one) into plugins_dir().

    Its licenses still have to be accepted before it can be chosen. An update
    (replace_existing) keeps the acceptance only if the license texts are
    unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="ez2d-plugin-") as scratch:
        if source.is_file() and zipfile.is_zipfile(source):
            folder = _unzip(source, Path(scratch))
        elif source.is_dir():
            folder = source
        else:
            raise PluginError(f"{source}: not a plugin folder or .zip")
        plugin = read_plugin(folder)
        target_root = plugins_dir()
        target = target_root / plugin.id
        if target.exists() and not replace_existing:
            raise PluginError(f"plugin {plugin.id} is already installed; remove or update it")
        target_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{plugin.id}.", dir=target_root))
        try:
            shutil.copytree(folder, staging, dirs_exist_ok=True, symlinks=True)
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return read_plugin(target)


def _unzip(archive: Path, scratch: Path) -> Path:
    """Extract safely (no paths outside, exec bits kept); the folder with the manifest."""
    out = scratch / "unpacked"
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            # Backslashes and drive letters would leave the folder on Windows.
            if name.is_absolute() or ".." in name.parts or any(c in info.filename for c in ":\\"):
                raise PluginError(f"{archive}: unsafe path {info.filename!r} in the archive")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise PluginError(f"{archive}: symbolic links are not allowed in a plugin .zip")
            path = out / name
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if (info.external_attr >> 16) & 0o111:
                path.chmod(0o755)
    if (out / MANIFEST).is_file():
        return out
    inner = [p for p in out.iterdir() if p.is_dir()] if out.is_dir() else []
    if len(inner) == 1 and (inner[0] / MANIFEST).is_file():
        return inner[0]
    raise PluginError(f"{archive}: no {MANIFEST} at its top level")


def remove(plugin_id: str) -> None:
    target = plugins_dir() / plugin_id
    if not ID_PATTERN.match(plugin_id) or not (target / MANIFEST).is_file():
        raise PluginError(f"no plugin {plugin_id!r} is installed")
    shutil.rmtree(target)
    state = _state()
    state["accepted"].pop(plugin_id, None)
    state["use"] = {k: v for k, v in state["use"].items() if v != plugin_id}
    _save_state(state)


# --- acceptance and choice ---------------------------------------------------------


def _state() -> dict[str, Any]:
    try:
        data = read_json_object(plugins_dir() / STATE_FILE)
    except FormatError:
        data = {}
    accepted = data.get("accepted")
    use = data.get("use")
    return {
        "accepted": dict(accepted) if isinstance(accepted, dict) else {},
        "use": dict(use) if isinstance(use, dict) else {},
    }


def _save_state(state: dict[str, Any]) -> None:
    folder = plugins_dir()
    folder.mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder / STATE_FILE, {"schema_version": 1, **state})


def accepted(plugin: Plugin) -> bool:
    return bool(_state()["accepted"].get(plugin.id) == plugin.license_digest())


def accept(plugin: Plugin) -> None:
    """Record that the user read and accepted the plugin's licenses as they are now."""
    state = _state()
    state["accepted"][plugin.id] = plugin.license_digest()
    _save_state(state)


def choose(slot: Slot, plugin: Plugin | None) -> None:
    """Use `plugin` for `slot` from now on; None: the built-in backend."""
    state = _state()
    if plugin is None:
        state["use"].pop(slot, None)
    else:
        if plugin.slot != slot:
            raise PluginError(f"{plugin.name} provides {plugin.slot}, not {slot}")
        if not accepted(plugin):
            raise PluginError(f"accept {plugin.name}'s licenses before using it")
        state["use"][slot] = plugin.id
    _save_state(state)


def chosen_id(slot: Slot) -> str | None:
    value = _state()["use"].get(slot)
    return value if isinstance(value, str) else None


def chosen(slot: Slot) -> Plugin | None:
    """The plugin to use for `slot`, None for the built-in backend.

    Raises PluginError if one was chosen but can't be used (removed, broken,
    licenses changed and not accepted again, not for this system): falling
    back to the built-in backend silently would give a different result.
    """
    plugin_id = chosen_id(slot)
    if plugin_id is None:
        return None
    advice = "install it again, or choose another in Settings → Plugins (ez2d plugins use)"
    try:
        plugin = read_plugin(plugins_dir() / plugin_id)
    except PluginError as exc:
        raise PluginError(f"the {slot} plugin {plugin_id} can't be used: {exc}; {advice}") from exc
    if not accepted(plugin):
        raise PluginError(
            f"{plugin.name}'s licenses changed since they were accepted; read and accept "
            "them again in Settings → Plugins (ez2d plugins accept)"
        )
    if not plugin.runs_here:
        raise PluginError(f"{plugin.name} doesn't run on this system; {advice}")
    return plugin


# --- running -------------------------------------------------------------------------


def pose_stage(
    plugin: Plugin,
    project: Project,
    bundles: Sequence[CaptureBundle],
    *,
    sfm: colmap.Colmap,
    masks: Path | None = None,
    stage: str = "mapping",
) -> StageSpec:
    """The camera placement plugin as the mapping stage: `<stage>/sparse/0`.

    It gets the photos as COLMAP would: `{captures}` and the names in
    `{image_list}` (`<capture id>/IMG_0001.jpg`), and with `with_masks`, a
    mask for every photo in `{masks}/<name>.png` (black: leave out).
    """
    stage_dir = project.stage_dir(stage)
    names = colmap.image_names(bundles)
    masked = masks is not None and bool(plugin.with_masks)
    values = {
        "colmap": str(sfm.path),
        "captures": str(project.captures_dir),
        "image_list": str(stage_dir / IMAGE_LIST),
        "masks": str(stage_dir / MASKS_DIR),
    }
    argv = _argv(plugin, plugin.command, stage_dir, values)
    if masked:
        argv += _expand(plugin.with_masks, plugin, stage_dir, values)
    inputs = {"captures": fingerprint([capture_input(b) for b in bundles])}
    if masked and masks is not None:
        inputs["masks"] = tree_input(masks)

    def prepare(folder: Path) -> None:
        (folder / "sparse").mkdir()
        (folder / IMAGE_LIST).write_text("\n".join(names) + "\n", encoding="utf-8")
        if masked and masks is not None:
            colmap.stage_masks(project, masks, names, folder / MASKS_DIR)

    return _spec(plugin, stage, argv, inputs, {"masked": masked}, prepare)


def splat_stage(
    plugin: Plugin,
    project: Project,
    undistorted: StageManifest,
    *,
    sfm: colmap.Colmap,
    options: brush.SplatOptions | None = None,
    masks: StageManifest | None = None,
    stage: str = "splat",
) -> StageSpec:
    """The splat plugin as the splat stage: `{dataset}` in, `<stage>/splat.ply` out.

    The dataset is the one Brush gets (see backends.brush), masks included
    as `images/masks/<stem>.png` when the project has them.
    """
    options = options or brush.SplatOptions()
    stage_dir = project.stage_dir(stage)
    values = {
        "colmap": str(sfm.path),
        "dataset": str(stage_dir / "dataset"),
        "steps": str(options.total_steps),
        "max_resolution": str(options.max_resolution),
        "max_splats": str(options.max_splats),
        "sh_degree": str(options.sh_degree),
    }
    argv = _argv(plugin, plugin.command, stage_dir, values)
    inputs = {"undistorted": stage_input(undistorted)}
    if masks is not None:
        inputs["masks"] = stage_input(masks)
    parameters = {
        "total_steps": options.total_steps,
        "max_resolution": options.max_resolution,
        "max_splats": options.max_splats,
        "sh_degree": options.sh_degree,
        "masked": masks is not None,
    }
    prepare = brush.dataset_preparer(project, undistorted, masks)
    return _spec(plugin, stage, argv, inputs, parameters, prepare)


def _spec(
    plugin: Plugin,
    stage: str,
    argv: list[str | Path],
    inputs: dict[str, str],
    parameters: dict[str, Any],
    prepare: Callable[[Path], None],
) -> StageSpec:
    return StageSpec(
        name=stage,
        backend=plugin.backend,
        argv=argv,
        # The manifest: a plugin run through an interpreter (python run.py)
        # has the interpreter as its executable, whose hash says little.
        parameters={**parameters, "plugin_manifest": f"sha256:{sha256_file(plugin.manifest)}"},
        inputs=inputs,
        env=user_environment(),
        parse_line=PluginProgress(plugin) if plugin.progress else None,
        use_pty=plugin.pty,
        prepare=prepare,
        gpu=plugin.gpu,
    )


def user_environment() -> dict[str, str] | None:
    """Environment changes that give a plugin this computer's libraries, not the app's.

    A packaged app (PyInstaller) puts its own folder first in the library
    path and keeps the original in `<variable>_ORIG`; a plugin's programs
    (its Python, PyTorch...) must not load the app's copies. None when
    nothing needs changing.
    """
    if not getattr(sys, "frozen", False):
        return None
    env = {
        var: os.environ.get(f"{var}_ORIG", "")
        for var in ("LD_LIBRARY_PATH", "LIBPATH")
        if var in os.environ
    }
    return env or None


def _argv(
    plugin: Plugin, args: Sequence[str], stage_dir: Path, values: Mapping[str, str]
) -> list[str | Path]:
    expanded = _expand(args, plugin, stage_dir, values)
    program = expanded[0]
    path = Path(program)
    if not path.is_absolute() and len(path.parts) > 1:
        path = plugin.folder / path  # relative to the plugin: bin/run, ./run.sh
    # A bare name (python3, docker...) is looked up on PATH.
    found = executable(path) if len(path.parts) > 1 else find_tool(program)
    if found is None:
        raise PluginError(
            f"{plugin.name}: can't run {program!r} (not found, or not executable); "
            "see the plugin's instructions"
        )
    return [found, *expanded[1:]]


def _expand(
    args: Sequence[str], plugin: Plugin, stage_dir: Path, values: Mapping[str, str]
) -> list[str]:
    every = {
        "plugin": str(plugin.folder),
        "output": str(stage_dir),
        "threads": str(cpu_threads()),
        **values,
    }
    return [arg.format_map(every) for arg in args]


class PluginProgress:
    """Progress from the plugin's output, by its `[progress]` pattern."""

    def __init__(self, plugin: Plugin) -> None:
        self.pattern = re.compile(plugin.progress or "")
        self.message = plugin.progress_message or f"Running {plugin.name}"

    def __call__(self, line: str) -> Progress | None:
        match = self.pattern.search(line)
        if match is None:
            return None
        groups = match.groupdict()
        try:
            if "done" in groups and "total" in groups:
                done, total = float(groups["done"]), float(groups["total"])
            else:
                done, total = float(match.group(1)), float(match.group(2))
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        return Progress(self.message, round(min(max(done / total, 0.0), 1.0), 4))


def describe(plugin: Plugin) -> str:
    """One line for notices and reports: name, version, licenses."""
    free = "" if plugin.free else "; not known to be a free license: check its terms"
    return f"{plugin.label()} (plugin; {plugin.license_summary()}{free})"


def report() -> dict[str, Any]:
    """Installed plugins and the choices, for diagnostics."""
    found = installed()
    return {
        "dir": str(plugins_dir()),
        "installed": {
            p.id: {
                "version": p.version,
                "provides": p.slot,
                "licenses": p.license_summary(),
                "accepted": accepted(p),
            }
            for p in found.plugins
        },
        "use": {slot: chosen_id(slot) for slot in SLOTS},
        "problems": list(found.problems),
    }
