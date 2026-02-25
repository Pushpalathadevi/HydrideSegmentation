# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MicroSeg Qt desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

repo_root = Path(__file__).resolve().parents[3]

hiddenimports = collect_submodules("hydride_segmentation")
hiddenimports += collect_submodules("src.microseg")

datas = [
    (str(repo_root / "data" / "sample_images"), "data/sample_images"),
    (str(repo_root / "frozen_checkpoints" / "model_registry.json"), "frozen_checkpoints"),
    (str(repo_root / "configs"), "configs"),
]

a = Analysis(
    [str(repo_root / "hydride_segmentation" / "qt_gui.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MicroSegDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

