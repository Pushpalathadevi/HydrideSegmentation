# Desktop App Workspace

Current v1.0.0 implementation status:

- Phase 2 workflow manager is in `src/microseg/app/desktop_workflow.py`.
- Phase 3 introduces Qt GUI foundation:
  - `hydride_segmentation/qt_gui.py`
  - `hydride_segmentation/qt/main_window.py`

Current default GUI direction:
- Qt (`PySide6`) is the primary framework for advanced correction workflows.
- Tkinter GUI remains available as compatibility path (`--framework tk`).

Windows packaging assets:
- PyInstaller spec: `windows/microseg_desktop.spec`
- Inno Setup script: `windows/microseg_desktop.iss`
- build helper script: `../../scripts/build_windows_installer.ps1`

The build produces an application folder containing `MicroSegDesktop.exe` and
`MicroSegCLI.exe`, then wraps that folder into one per-user offline setup file.
Installed runtime data is synchronized into the writable local application-data
workspace so logs, outputs, local registry overlays, and checkpoints never need
write access to the installation directory.
