# Developer Guide

This guide covers day-to-day development against the current baseline while the repository transitions to a general microstructural segmentation platform.

## Start Here

- Mission and scope: `docs/mission_statement.md`
- Baseline audit: `docs/base_zero_audit.md`
- Target architecture: `docs/target_architecture.md`
- Phase plan: `docs/development_roadmap.md`
- Working rules: `AGENTS.md`

## Local Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Existing Entry Points

```bash
hydride-gui
hydride-gui-qt
hydride-orientation --help
segmentation-eval --help
package-corrections-dataset --help
```

Qt runtime note:
- `hydride-gui` defaults to Qt framework and requires `PySide6`.
- Legacy Tk path remains available via `hydride-gui --framework tk`.

## Development Procedure

Follow `docs/development_workflow.md` for implementation order, testing expectations, and documentation sync requirements.

## Phase 1 Core Entry Points

- Microseg pipeline: `src/microseg/pipelines/segmentation_pipeline.py`
- Hydride compatibility adapter: `hydride_segmentation/microseg_adapter.py`
- Core contracts: `src/microseg/domain/contracts.py`

## Phase 2 Desktop Workflow Entry Points

- Desktop workflow manager: `src/microseg/app/desktop_workflow.py`
- GUI integration: `hydride_segmentation/core/gui_app.py`
- GUI model and processing helpers: `hydride_segmentation/core/image_processing.py`

Current GUI capabilities:
- registry-backed model selection
- single-image and batch execution
- run history browsing
- structured run export with `manifest.json` and `metrics.json`

## Phase 3 Correction and Export Entry Points

- Correction session utilities: `src/microseg/corrections/session.py`
- Correction export and dataset packaging: `src/microseg/corrections/exporter.py`
- Qt desktop app: `hydride_segmentation/qt/main_window.py`
- Correction dataset CLI: `scripts/package_corrections_dataset.py`

Correction UX controls implemented:
- zoom in/out/reset (+ Ctrl-wheel zoom)
- predicted/corrected/diff layer toggles and transparency sliders
- polygon and lasso interactive tools
- split-view synchronized pan/zoom correction workspace
- keyboard shortcuts for tool selection and undo/redo
