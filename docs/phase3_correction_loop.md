# Phase 3 - Human Correction and Data Export Loop

Date: 2026-02-15
Branch: `codex/microstructure-foundation-scaffold`

## Scope Delivered

1. Correction session model
- Added `CorrectionSession` with class-index aware brush/polygon actions and undo/redo.
- Added connected-feature operations: delete and relabel.
- Module: `src/microseg/corrections/session.py`

2. Versioned correction export schema
- Schema version: `microseg.correction.v1`
- Record contract added in `src/microseg/domain/corrections.py`
- Export logic in `src/microseg/corrections/exporter.py`

3. Dataset packaging CLI
- CLI script: `scripts/package_corrections_dataset.py`
- Builds deterministic `train/val/test` layout with dataset manifest.

4. Qt-based GUI foundation for correction workflow
- Qt entry point: `hydride_segmentation/qt_gui.py`
- Main window: `hydride_segmentation/qt/main_window.py`
- Features:
  - model selection from registry-backed workflow manager
  - segmentation run (single + batch)
  - brush, polygon, and lasso correction tools
  - feature-select correction tool for connected component delete/relabel
  - class map editing and active class selection
  - zoom controls + Ctrl-wheel zoom
  - layer visibility/transparency controls (predicted/corrected/diff)
  - split-view synchronized pan/zoom workspace
  - undo/redo correction
  - corrected sample export through versioned schema with selectable formats
  - project session save/load for restartable annotation work
  - workflow hub for deterministic dataset split packaging

## Exported Corrected Sample Layout

Each correction sample folder contains:
- `input.png`
- `predicted_mask_indexed.png`
- `corrected_mask_indexed.png`
- `predicted_mask_color.png` (optional)
- `corrected_mask_color.png` (optional)
- `corrected_mask.npy` (optional)
- `corrected_overlay.png`
- `correction_record.json`

`correction_record.json` includes:
- schema version
- source/run/model provenance
- annotator and notes
- metrics snapshot
- class map used for annotation
- export format manifest
- corrected/predicted foreground counts

## Dataset Packaging Output Layout

```
<output_dir>/
  dataset_manifest.json
  train/
    images/
    masks/
    metadata/
  val/
    images/
    masks/
    metadata/
  test/
    images/
    masks/
    metadata/
```

## Validation Status

- Added tests: `tests/test_phase3_corrections.py`
- Added tests: `tests/test_phase3_annotation_view.py`
- Full suite: `pytest -q` -> passing on CPU-only setup

## Current Limitations

- Qt GUI requires `PySide6` at runtime.
- Correction tools currently use raster overlays without edge-snapping assistance.
- End-to-end model training orchestration remains a Phase 4+ objective.
