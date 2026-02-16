# HydrideSegmentation -> Microstructural Segmentation Platform (Transition)

Current release version: `0.5.0`

This repository is transitioning from a hydride-specific toolkit into a general-purpose microstructural segmentation platform.

Hydride segmentation remains the baseline implemented workflow.
The long-term mission is broader, model-agnostic segmentation and analysis for microscopy images with local desktop deployment.

## Mission

Build a scientifically robust local application and backend stack for microstructural segmentation with:
- pluggable segmentation models
- quantitative analysis pipelines
- human-in-the-loop correction
- correction export for future model retraining

See `docs/mission_statement.md`.

## Current Baseline (Base Zero)

The existing repository already provides:
- conventional hydride segmentation
- ML-based hydride segmentation inference
- Tkinter GUI for local usage
- orientation and size analysis
- Flask API endpoints
- dataset preparation and augmentation scripts

Baseline audit: `docs/base_zero_audit.md`

## Transition Roadmap and Architecture

- Docs index: `docs/README.md`
- Target architecture: `docs/target_architecture.md`
- Phase plan: `docs/development_roadmap.md`
- Repository blueprint: `docs/repository_blueprint.md`
- Migration strategy: `docs/migration_strategy.md`
- Scientific validation protocol: `docs/scientific_validation.md`
- Local app product spec: `docs/local_desktop_product_spec.md`
- Versioning/release policy: `docs/versioning_and_release_policy.md`

## Phase 1 Progress

The new model-agnostic orchestration foundation is now in place under `src/microseg` with:
- domain contracts
- predictor/analyzer interfaces
- model registry
- hydride adapters
- compatibility pipeline bridge used by the GUI segmentation path

## Phase 2 Progress

Desktop app workflow has been refactored around `src/microseg`:
- model selection now uses registry metadata
- batch processing is available in GUI
- run history browsing is available in GUI
- save/export now writes structured run packages (`manifest.json`, `metrics.json`, image artifacts)

Implementation notes: `docs/phase2_desktop_refactor.md`

## Phase 3 Progress

Human-correction loop and export pipeline are now implemented:
- correction session model with undo/redo (`src/microseg/corrections/session.py`)
- versioned correction export schema `microseg.correction.v1`
- correction dataset packaging CLI (`scripts/package_corrections_dataset.py`)
- Qt GUI foundation for correction workflows (`hydride_segmentation/qt_gui.py`)
- correction UX controls: zoom, layer transparency, diff overlays, shortcuts
- interactive tools: brush, polygon, and lasso
- split-view synchronized pan/zoom correction workspace

Implementation notes: `docs/phase3_correction_loop.md`
GUI UX details: `docs/phase3_correction_gui_features.md`

## Existing Usage (Still Supported)

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Run GUI (Qt default):
```bash
hydride-gui
```

Optional legacy Tk GUI:
```bash
hydride-gui --framework tk
```

3. Run orientation analysis from CLI:
```bash
hydride-orientation --image test_data/3PB_SRT_data_generation_1817_OD_side1_8.png --shouldRunSegmentation
```

4. Run segmentation evaluator:
```bash
segmentation-eval --simulate --plot
```

5. Run minimal API app:
```bash
python -m hydride_segmentation.app_minimal.app
```

6. Package corrected exports into train/val/test dataset:
```bash
python scripts/package_corrections_dataset.py --input-dir <correction_exports> --output-dir <dataset_out>
```

## Model Weights

ML inference expects model weights from:
- `HYDRIDE_MODEL_PATH` environment variable, or
- `/opt/models/hydride_segmentation/model.pt` by default.

## GUI Dependency Note

Qt GUI requires `PySide6` runtime.
Install with:
```bash
pip install PySide6
```

## Contributing

- Contributor guide: `CONTRIBUTE.md`
- Repository rules for humans and agents: `AGENTS.md`

## License

MIT (see `LICENSE`).
