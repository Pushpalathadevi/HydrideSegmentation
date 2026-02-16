# microseg Core Package

This package is the target architecture for model-agnostic microstructural segmentation.

Current status:
- Phase 1 baseline implemented:
  - domain contracts (`domain/contracts.py`)
  - predictor/analyzer interfaces (`core/interfaces.py`)
  - model registry (`plugins/registry.py`)
  - hydride predictor adapters (`inference/predictors.py`)
  - segmentation orchestration (`pipelines/segmentation_pipeline.py`)
  - compatibility bridge (`hydride_segmentation/microseg_adapter.py`)
- Phase 2 desktop workflow layer implemented:
  - `app/desktop_workflow.py` for single/batch execution, history, and export packages
- Phase 3 correction and export loop implemented:
  - `corrections/session.py` for mask editing session + undo/redo
  - `corrections/exporter.py` for schema-based corrected sample export and dataset packaging
- Existing implementation still remains under `hydride_segmentation/` for backward compatibility.
