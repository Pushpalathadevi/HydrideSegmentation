"""Release-contract tests for MicroSeg v1.0.0 desktop packaging."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from src.microseg.app.orchestration import OrchestrationCommandBuilder
from src.microseg.app.runtime_paths import prepare_desktop_workspace, resolve_desktop_runtime_paths
from src.microseg.version import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent_across_public_metadata() -> None:
    """Every user-visible packaging surface must publish the same version."""

    assert __version__ == "1.0.0"
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__

    setup_text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert re.search(r"version=['\"]1\.0\.0['\"]", setup_text)

    for relative in (
        "README.md",
        "docs/versioning_and_release_policy.md",
        "apps/desktop/windows/microseg_desktop.iss",
    ):
        assert "1.0.0" in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_cli_exposes_release_version() -> None:
    """The source CLI must provide a machine-checkable release version."""

    completed = subprocess.run(
        [sys.executable, "scripts/microseg_cli.py", "--version"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"MicroSeg {__version__}"


def test_packaging_spec_builds_gui_and_cli_in_one_application_folder() -> None:
    """The installer input must contain both launchers and bundled defaults."""

    spec = (REPO_ROOT / "apps/desktop/windows/microseg_desktop.spec").read_text(encoding="utf-8")
    assert 'name="MicroSegDesktop"' in spec
    assert 'name="MicroSegCLI"' in spec
    assert "COLLECT(" in spec
    assert "Path(SPECPATH).resolve().parents[2]" in spec
    assert "Path(__file__)" not in spec
    assert 'collect_submodules("scripts")' not in spec
    assert '"windows" / "hooks"' in spec
    assert '"configs"' in spec
    assert '"data/sample_images"' in spec
    assert '"frozen_checkpoints"' in spec

    qt_network_hook = (
        REPO_ROOT / "apps/desktop/windows/hooks/hook-PySide6.QtNetwork.py"
    ).read_text(encoding="utf-8")
    assert "add_qt6_dependencies" in qt_network_hook
    assert "collect_qtnetwork_files" not in qt_network_hook


def test_installer_is_per_user_single_file_and_uses_built_application_folder() -> None:
    """Inno Setup must emit one per-user installer from the onedir build."""

    installer = (REPO_ROOT / "apps/desktop/windows/microseg_desktop.iss").read_text(encoding="utf-8")
    assert "OutputBaseFilename=MicroSegDesktop_{#AppVersion}_offline_setup" in installer
    assert "PrivilegesRequired=lowest" in installer
    assert "DefaultDirName={localappdata}\\Programs\\{#AppName}" in installer
    assert 'Source: "{#DistRoot}\\*"' in installer

    build_script = (REPO_ROOT / "scripts/build_windows_installer.ps1").read_text(encoding="utf-8")
    assert "installed_smoke.json" in build_script
    assert "Invoke-CheckedGuiProcess" in build_script
    assert "Start-Process" in build_script and "-Wait" in build_script
    assert "Remove-Item -LiteralPath $smokeReport -Force" in build_script
    assert "packagedSmoke.app_version -ne $version" in build_script
    assert "installedSmoke.app_version -ne $version" in build_script
    assert "Installed desktop smoke report was not created" in build_script


def test_workspace_sync_refreshes_defaults_without_overwriting_local_models(tmp_path: Path) -> None:
    """Installed upgrades must preserve local overlays and checkpoint binaries."""

    resources = tmp_path / "resources"
    workspace = tmp_path / "workspace"
    (resources / "configs" / "app").mkdir(parents=True)
    (resources / "data" / "sample_images").mkdir(parents=True)
    (resources / "frozen_checkpoints").mkdir(parents=True)
    (resources / "pre_trained_weights").mkdir(parents=True)
    (resources / "configs" / "app" / "desktop_ui.default.yml").write_text("version: new\n", encoding="utf-8")
    (resources / "data" / "sample_images" / "sample.png").write_bytes(b"sample")
    (resources / "frozen_checkpoints" / "model_registry.json").write_text("{}\n", encoding="utf-8")
    (resources / "pre_trained_weights" / "registry.template.json").write_text("{}\n", encoding="utf-8")

    local_overlay = workspace / "frozen_checkpoints" / "model_registry.local.json"
    local_binary = workspace / "frozen_checkpoints" / "candidates" / "local.pt"
    local_overlay.parent.mkdir(parents=True)
    local_binary.parent.mkdir(parents=True)
    local_overlay.write_text('{"local": true}\n', encoding="utf-8")
    local_binary.write_bytes(b"checkpoint")

    initialized = prepare_desktop_workspace(resources, workspace)

    assert initialized == workspace.resolve()
    assert (workspace / "configs" / "app" / "desktop_ui.default.yml").read_text(encoding="utf-8") == "version: new\n"
    assert (workspace / "data" / "sample_images" / "sample.png").read_bytes() == b"sample"
    assert (workspace / "pre_trained_weights" / "registry.template.json").is_file()
    assert local_overlay.read_text(encoding="utf-8") == '{"local": true}\n'
    assert local_binary.read_bytes() == b"checkpoint"
    manifest = json.loads((workspace / "desktop_runtime.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "microseg.desktop_runtime.v1"
    assert manifest["app_version"] == __version__


def test_source_runtime_uses_repository_root() -> None:
    """Source launches must retain the existing repository workflow."""

    runtime = resolve_desktop_runtime_paths(start=REPO_ROOT / "src" / "microseg" / "app")
    assert runtime.frozen is False
    assert runtime.resource_root == REPO_ROOT
    assert runtime.workspace_root == REPO_ROOT
    assert runtime.cli_executable is None


def test_orchestrator_prefers_packaged_cli_companion(tmp_path: Path) -> None:
    """Installed GUI jobs must call the console companion directly."""

    cli = tmp_path / "MicroSegCLI.exe"
    builder = OrchestrationCommandBuilder(repo_root=tmp_path, cli_executable=cli)
    command = builder.infer(image="image.png", model_name="Hydride Conventional")
    assert command[:2] == [str(cli), "infer"]
    assert "scripts/microseg_cli.py" not in " ".join(command)
