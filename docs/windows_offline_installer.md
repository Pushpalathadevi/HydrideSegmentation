# Windows Offline Installer Workflow

This runbook creates a single offline installer `.exe` for the Qt desktop app.

## Prerequisites

- Windows 10/11 build machine
- Python 3.10+
- Repository checkout
- Inno Setup 6 (`ISCC.exe`) for single-installer packaging; the build script
  checks `PATH` and the standard machine/per-user installation locations

## 1. Install Dependencies

```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-build.txt
python -m pip install -e .
```

## 2. Build Desktop Executable

PyInstaller spec:
- `apps/desktop/windows/microseg_desktop.spec`

Manual build command:

```powershell
python -m PyInstaller --noconfirm --clean apps/desktop/windows/microseg_desktop.spec
```

Output:
- `dist/MicroSegDesktop/MicroSegDesktop.exe` (windowed GUI)
- `dist/MicroSegDesktop/MicroSegCLI.exe` (console/background-job companion)

Bundled assets include:
- `data/sample_images/`
- `frozen_checkpoints/model_registry.json`
- `configs/`
- `pre_trained_weights/` metadata and registry template

The installed application synchronizes canonical defaults into
`%LOCALAPPDATA%\MicroSeg\workspace`. This writable workspace owns logs, outputs,
local model overlays, and user-installed checkpoints; the application never
requires write access to its installation directory.

## 3. Build Single Offline Installer `.exe`

Inno Setup script:
- `apps/desktop/windows/microseg_desktop.iss`

Compile installer:

```powershell
iscc apps/desktop/windows/microseg_desktop.iss
```

Installer output:
- `dist/installer/MicroSegDesktop_1.0.0_offline_setup.exe`

Release evidence:

- `dist/installer/MicroSegDesktop_1.0.0_release.json`
- SHA-256 recorded in that JSON manifest
- packaged smoke report under `dist/`
- installed smoke report and Inno install log under
  `dist/installer/verification/`

## 4. One-Command Script

Use the provided script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1
```

Options:
- `-InstallBuildDependencies` installs the pinned `requirements-build.txt`
  toolchain when PyInstaller is missing.
- `-SkipTests` skips the release/desktop unit smoke subset.
- `-SkipInstaller` builds and smoke-tests only the PyInstaller application.
- `-SkipPackagedSmokeTest` skips packaged GUI/CLI launch checks.
- `-SkipInstallerVerification` skips silent installation, installed-app smoke,
  and uninstall checks.

The two verification-skip switches are for build debugging only. Do not publish
an artifact produced with either verification disabled.

## 5. Offline Validation Checklist

The build script performs an automated silent install/smoke/uninstall cycle in a
temporary verification folder. Before public distribution, also validate on a
clean Windows 10/11 target machine:

1. Install using the generated `.exe`.
2. Launch `MicroSeg Desktop`.
3. Load a bundled sample image (`File -> Open Sample`).
4. Run segmentation with `Hydride Conventional`.
5. Confirm `Results Dashboard` populates.
6. Export a results package and verify:
   - `results_summary.json`
   - `results_report.html`
   - `results_report.pdf`
7. Confirm log output under `outputs/logs/desktop/`.

For installed builds, the log path is under
`%LOCALAPPDATA%\MicroSeg\workspace\outputs\logs\desktop\`.

## Release limitation

The v1.0.0 setup executable is not Authenticode-signed. Windows can therefore
show an unknown publisher warning. Signing requires a trusted code-signing
certificate and a protected signing pipeline and is tracked as a post-v1.0.0
distribution-hardening item.
