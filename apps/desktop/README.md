# Desktop App Workspace

Current implementation status:

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

Planned migration:
- Move Qt desktop app assembly into `apps/desktop/` package boundaries while preserving stable CLI entry points.
