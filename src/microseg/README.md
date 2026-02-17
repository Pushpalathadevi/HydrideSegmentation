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
  - `corrections/classes.py` for class index/color map contracts
  - `corrections/exporter.py` for schema-based corrected sample export and dataset packaging
- Phase 4 foundation scaffolding implemented:
  - `app/project_state.py` for project/session persistence
  - `app/orchestration.py` for infer/train/evaluate/package command construction
  - `io/configuration.py` for YAML + `--set` override resolution
- Phase 4 orchestration/training baseline implemented:
  - `training/pixel_classifier.py` CPU baseline training/inference
  - `training/torch_pixel_classifier.py` GPU-compatible torch baseline training/inference with CPU fallback
  - `training/unet_binary.py` UNet binary training with checkpoints, early stopping, resume, and tracked val sample reporting
  - `evaluation/pixel_model_eval.py` baseline evaluation with tracked sample panels + HTML summary output
- Phase 7 observability and model metadata implemented:
  - `plugins/frozen_checkpoints.py` frozen-checkpoint metadata registry loader
  - `frozen_checkpoints/model_registry.json` metadata source for model guidance
  - training/evaluation JSON + HTML report emission with progress/ETA logging
- Phase 8 quality governance implemented:
  - `quality/phase_gate.py` phase closeout checks and stocktake report generation
- Phase 9 model lifecycle + dataops foundation implemented:
  - `plugins/registry_validation.py` strict frozen checkpoint metadata validator
  - `dataops/split_planner.py` leakage-aware correction split materialization
  - `dataops/quality.py` packaged dataset QA checks
- Phase 10 training data contract implemented:
  - `dataops/training_dataset.py` split-layout detection + unsplit auto-prepare with ID suffix mapping
- Phase 11 dataset policy alignment implemented:
  - `dataops/training_dataset.py` leakage-aware default split strategy + optional RGB colormap conversion
- Phase 12 GUI dataset workspace foundation implemented:
  - `dataops/training_dataset.py` preview API (`preview_training_dataset_layout`)
  - `app/orchestration.py` dataset-prepare and dataset-qa command builders
- Existing implementation still remains under `hydride_segmentation/` for backward compatibility.
