# Phase 33 Closeout: Interactive Conventional Segmentation GUI

## Goal and delivered behavior

The Qt desktop GUI now provides an interactive conventional-segmentation loop. The `Segmentation` tab shows input and predicted mask side by side. Conventional controls sit above the viewers, contain canonical defaults, and explain their effect through hover tooltips. Parameter edits use a 400 ms debounce and the existing background worker.

Loading a new image clears the previous run association and result pane. The pane reads `Result not ready yet` until inference produces a mask and `Updating result` while a live refresh is queued. If parameters change during a running job, one refresh is retained and starts after completion.

## Verification and traceability

- Not-ready screenshot: `artifacts/screenshots/qt_gui_conventional_live_not_ready_v033.png`
- Completed-result screenshot: `artifacts/screenshots/qt_gui_conventional_live_result_v033.png`
- Reproducible capture helper: `scripts/capture_phase33_conventional_gui.py`
- New GUI regression module: `tests/test_phase33_interactive_conventional_gui.py`
- Existing Qt regression module: `tests/test_phase27_qt_settings_smoke.py`
- Conventional workflow/regression coverage: `tests/test_phase2_desktop_workflow.py`, `tests/test_phase0_regression.py`, and `tests/test_core.py`
- Machine-readable closeout: `docs/phase33_interactive_conventional_gui.report.json`

## Remaining gaps

- Automated screenshot comparison is not yet a release gate; Qt structure and behavior are covered offscreen.
- Very large images rerun the full conventional pipeline rather than a downsampled preview followed by full-resolution confirmation.
- Existing scikit-image morphology deprecation warnings remain outside this GUI phase.
