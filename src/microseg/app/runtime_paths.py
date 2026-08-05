"""Runtime path discovery for source and installed desktop deployments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

from src.microseg.version import __version__


RUNTIME_MANIFEST_SCHEMA = "microseg.desktop_runtime.v1"


@dataclass(frozen=True)
class DesktopRuntimePaths:
    """Resolved resource, workspace, and CLI paths for the desktop app.

    Parameters
    ----------
    resource_root:
        Read-only root containing assets bundled with the application.
    workspace_root:
        Writable root used for logs, outputs, local registry overlays, and
        user-installed checkpoints.
    cli_executable:
        Optional packaged CLI companion. Source checkouts use the active
        Python interpreter plus ``scripts/microseg_cli.py`` instead.
    frozen:
        Whether the application is running from a PyInstaller bundle.
    """

    resource_root: Path
    workspace_root: Path
    cli_executable: Path | None
    frozen: bool


def _source_repo_root(start: Path) -> Path:
    """Locate the source checkout containing the CLI and default configs."""

    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "scripts" / "microseg_cli.py").exists() and (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError("could not locate repository root containing scripts/microseg_cli.py")


def _user_data_root() -> Path:
    """Return a writable per-user application-data root."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "MicroSeg"


def _copy_tree_contents(source: Path, destination: Path) -> None:
    """Copy a bundled directory into the writable runtime workspace."""

    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_files(resource_root: Path, workspace_root: Path, relative_paths: Iterable[str]) -> None:
    """Synchronize individual bundled files into the runtime workspace."""

    for relative in relative_paths:
        source = resource_root / relative
        if not source.is_file():
            continue
        destination = workspace_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def prepare_desktop_workspace(resource_root: Path, workspace_root: Path) -> Path:
    """Create or refresh the installed desktop application's writable workspace.

    Bundled canonical configuration and metadata are refreshed on every launch
    so they stay synchronized with the installed application version. Local
    registry overlays and checkpoint binaries are deliberately left untouched.

    Parameters
    ----------
    resource_root:
        PyInstaller resource root (normally ``sys._MEIPASS``).
    workspace_root:
        Per-user destination root.

    Returns
    -------
    pathlib.Path
        The initialized writable workspace root.
    """

    resource_root = Path(resource_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    _copy_tree_contents(resource_root / "configs", workspace_root / "configs")
    _copy_tree_contents(resource_root / "data" / "sample_images", workspace_root / "data" / "sample_images")
    _copy_tree_contents(resource_root / "pre_trained_weights", workspace_root / "pre_trained_weights")
    _copy_files(
        resource_root,
        workspace_root,
        ("frozen_checkpoints/model_registry.json",),
    )

    for relative in (
        "frozen_checkpoints/candidates",
        "frozen_checkpoints/promoted",
        "frozen_checkpoints/smoke",
        "outputs/logs/desktop",
        "outputs/inference",
    ):
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA,
        "app_version": __version__,
        "resource_root": str(resource_root),
        "workspace_root": str(workspace_root),
    }
    (workspace_root / "desktop_runtime.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return workspace_root


def resolve_desktop_runtime_paths(*, start: Path | None = None) -> DesktopRuntimePaths:
    """Resolve desktop paths for a source checkout or installed application.

    Parameters
    ----------
    start:
        Optional source-mode path used for repository discovery.

    Returns
    -------
    DesktopRuntimePaths
        Runtime paths suitable for desktop initialization and orchestration.
    """

    frozen = bool(getattr(sys, "frozen", False))
    if not frozen:
        root = _source_repo_root(start or Path(__file__))
        return DesktopRuntimePaths(
            resource_root=root,
            workspace_root=root,
            cli_executable=None,
            frozen=False,
        )

    resource_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    workspace_root = prepare_desktop_workspace(resource_root, _user_data_root() / "workspace")
    cli_name = "MicroSegCLI.exe" if os.name == "nt" else "MicroSegCLI"
    cli_executable = Path(sys.executable).resolve().with_name(cli_name)
    if not cli_executable.is_file():
        cli_executable = None

    return DesktopRuntimePaths(
        resource_root=resource_root,
        workspace_root=workspace_root,
        cli_executable=cli_executable,
        frozen=True,
    )
