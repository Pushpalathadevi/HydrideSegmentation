# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MicroSeg Qt desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# PyInstaller executes spec files without defining ``__file__``. ``SPECPATH``
# is the supported absolute directory containing this spec.
repo_root = Path(SPECPATH).resolve().parents[2]

hiddenimports = collect_submodules("hydride_segmentation")
hiddenimports += collect_submodules("src.microseg")

datas = [
    (str(repo_root / "data" / "sample_images"), "data/sample_images"),
    (str(repo_root / "frozen_checkpoints" / "model_registry.json"), "frozen_checkpoints"),
    (str(repo_root / "configs"), "configs"),
    (str(repo_root / "pre_trained_weights"), "pre_trained_weights"),
    (str(repo_root / "README.md"), "."),
]

a = Analysis(
    [str(repo_root / "hydride_segmentation" / "qt_gui.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(repo_root / "apps" / "desktop" / "windows" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

desktop_exe = EXE(
    pyz,
    a.scripts,
    [],
    name="MicroSegDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    exclude_binaries=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

cli_exe = EXE(
    pyz,
    a.scripts,
    [],
    name="MicroSegCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    exclude_binaries=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collect = COLLECT(
    desktop_exe,
    cli_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MicroSegDesktop",
)

