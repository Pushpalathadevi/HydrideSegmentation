# Windows Offline Installer Workflow

This runbook creates a single offline installer `.exe` for the Qt desktop app.

## Prerequisites

- Windows 10/11 build machine
- Python 3.10+
- Repository checkout
- Inno Setup 6 (`iscc` on `PATH`) for single-installer packaging

## 1. Install Dependencies

```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
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
- `dist/MicroSegDesktop/MicroSegDesktop.exe`

Bundled assets include:
- `data/sample_images/`
- `frozen_checkpoints/model_registry.json`
- `configs/`

## 3. Build Single Offline Installer `.exe`

Inno Setup script:
- `apps/desktop/windows/microseg_desktop.iss`

Compile installer:

```powershell
iscc apps/desktop/windows/microseg_desktop.iss
```

Installer output:
- `dist/installer/MicroSegDesktop_0.22.0_offline_setup.exe`

## 4. One-Command Script

Use the provided script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1
```

Options:
- `-SkipInstaller` builds only the PyInstaller executable.
- `-SkipSmokeTest` skips the quick `pytest` smoke test.

## 5. Offline Validation Checklist

On a clean target machine:

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
