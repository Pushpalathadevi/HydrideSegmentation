# Phase 2 Desktop Refactor - Implementation Status

Date: 2026-02-15
Branch: `codex/microstructure-foundation-scaffold`

## Scope Implemented

Phase 2 target: local desktop workflow refactor around the new `src/microseg` core.

Implemented components:

1. Model registry-driven model selection
- GUI model options now come from registry metadata instead of hardcoded backend map.
- Legacy model labels remain supported for compatibility.

2. Desktop workflow manager
- Added `DesktopWorkflowManager` (`src/microseg/app/desktop_workflow.py`) for:
  - single-run execution
  - batch-run execution
  - in-memory run history
  - structured run export

3. Run history browsing in GUI
- Added run history list in the Tkinter app.
- Selecting a history entry restores corresponding input/mask/overlay panels.

4. Batch run manager in GUI
- Added `Run Batch` action using multi-file selection.
- Batch progress is logged and last successful run is loaded into panels.

5. Structured export package
- `Save Results` now exports a run package directory containing:
  - `input.png`
  - `prediction.png`
  - `overlay.png`
  - analysis plot images (if available)
  - `metrics.json`
  - `manifest.json`
  - `results_panel.png` (captured GUI panel)

## Updated Modules

- `hydride_segmentation/core/gui_app.py`
- `hydride_segmentation/core/image_processing.py`
- `hydride_segmentation/microseg_adapter.py`
- `src/microseg/app/desktop_workflow.py`
- `src/microseg/app/__init__.py`

## Tests Added for Phase 2

- `tests/test_phase2_desktop_workflow.py`
  - model option discovery
  - single run + export package creation
  - batch execution and history recording

## Validation Result

`pytest -q` -> `14 passed` (CPU-only local run)

## Remaining Work for Later Phases

- Interactive correction tools and correction provenance schema (Phase 3)
- Training-data export pipeline for corrected annotations (Phase 3)
- Packaging/distribution hardening for desktop installers (Phase 5)
