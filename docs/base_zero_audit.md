# Base Zero Audit (Current Repository State)

Date: 2026-02-15
Branch baseline reviewed: `main`

This document captures what already works before refactor or rewrite.

## Existing Functional Capabilities

1. Conventional segmentation pipeline
- Module: `hydride_segmentation/segmentation_mask_creation.py`
- Uses CLAHE + adaptive threshold + morphology + area filtering
- Exposes `run_model(image_path, params)`

2. ML inference pipeline
- Module: `hydride_segmentation/inference.py`
- Uses `segmentation_models_pytorch.Unet(resnet18)`
- Loads weights from `HYDRIDE_MODEL_PATH` (default `/opt/models/hydride_segmentation/model.pt`)
- Exposes `run_model(image_path, params=None, weights_path=...)`

3. Desktop GUI
- Module: `hydride_segmentation/core/gui_app.py`
- Tkinter + drag-drop (`tkinterdnd2`)
- Supports model selection (conventional/ML), run, overlay visualization, zoom, undo/redo, save results
- Orientation/area-fraction analysis available from menu

4. Orientation and morphology analysis
- Modules:
  - `hydride_segmentation/core/analysis.py`
  - `hydride_segmentation/hydride_orientation_analyzer.py`
- Produces orientation map and histograms

5. API endpoints
- Blueprint API: `hydride_segmentation/api/*`
  - `/health`
  - `/segment`
- Legacy Flask service: `hydride_segmentation/service.py`

6. Dataset and augmentation utilities
- `hydride_segmentation/prepare_dataset.py`
- Scripts in `examples/` for image collection, augmentation, mask export, and synthetic composition

7. Basic test coverage
- `tests/test_core.py`
- `tests/test_api.py`

## Verified Gaps and Risks

1. Import-time coupling blocks tests in minimal environments
- `hydride_segmentation/__init__.py` imports GUI module directly.
- If `tkinterdnd2` is missing, even API tests fail during import.

2. Architecture overlap and duplication
- Multiple API styles (legacy + blueprint) and overlapping segmentation access paths.
- Conventional pipeline implemented in more than one style.

3. Limited reproducibility metadata
- Inference and analysis outputs do not consistently persist config hash, model ID, or run manifest.

4. Plugin/model generalization not yet formalized
- Current design is hydride-specific with ad hoc backend selection.

5. Human correction feedback loop not implemented
- No standard correction schema or export-to-training dataset path.

## Base-Zero Preservation Requirement

All existing workflows above are considered baseline features.
Future refactor work must either:
- keep behavior intact, or
- provide tested and documented equivalent replacements.
